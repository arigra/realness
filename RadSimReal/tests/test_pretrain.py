"""Reference RAD3D decoder/loss and provenance checks beyond Cartesian tests."""
import math
import pytest
import torch
from radsimreal.bev import create_model
from radsimreal.bev.rad_pretraining import RADDetRAD,decode_rad,rad_loss,iou_rad,load_pretrained_raddet_backbone


def test_rad_decoder_known_grid_center_size_and_score():
    output = {"raw":torch.zeros(1,2,2,1,1,8),"anchors":torch.tensor([[.2,.3,.4]]),"rad_shape":(32,32,4)}
    boxes,scores = decode_rad(output)
    assert torch.allclose(boxes[0,0,0,0,0],torch.tensor([.25,.25,.5,.2,.3,.4]))
    assert torch.allclose(boxes[0,1,1,0,0,:3],torch.tensor([.75,.75,.5]))
    assert scores.unique().tolist() == [.5]


def test_rad_source_loss_is_near_zero_for_exact_confident_prediction():
    raw = torch.zeros(1,2,2,1,1,8)
    raw[...,6] = -20
    raw[0,0,0,0,0,6:] = 20
    output = {"raw":raw.requires_grad_(),"anchors":torch.tensor([[.2,.3,.4]]),"rad_shape":(32,32,4)}
    target = [torch.tensor([[.25,.25,.5,.2,.3,.4]])]
    losses = rad_loss(output,target)
    assert losses["box"].item() == pytest.approx(0,abs=1e-12)
    assert losses["total"].item() < 1e-5
    losses["total"].backward()
    assert torch.isfinite(raw.grad).all()


def test_rad_overlap_known_volume_and_empty_cases():
    box = torch.tensor([[0.,0,0,1,1,1]])
    other = torch.tensor([[.5,0,0,1,1,1],[3,0,0,1,1,1]])
    assert iou_rad(box,other).tolist()[0] == pytest.approx([1/3,0])
    assert iou_rad(torch.empty(0,6),other).shape == (0,2)


def test_rad_backbone_transfer_rejects_coordinate_profile_changes(tmp_path):
    torch.set_num_threads(1)
    source = RADDetRAD((4,16,16))
    checkpoint = tmp_path/"pretrain.pt"
    torch.save({"format_version":"raddet_rad_v2","model_config":source.export_config,"model_state":source.state_dict(),
                "normalization":{"scale":1},"dataset_meta":{"native_profile":{"range_step":.2,"representation":"complex"}},
                "global_step":1},checkpoint)
    target = create_model("raddet",input_shape=(4,16,16))
    with pytest.raises(ValueError,match="axes/representation"):
        load_pretrained_raddet_backbone(target,checkpoint,normalization={"scale":1},
            dataset_meta={"native_profile":{"range_step":.4,"representation":"complex"}})
    with pytest.raises(ValueError,match="normalization"):
        load_pretrained_raddet_backbone(target,checkpoint,normalization={"scale":2})


def test_native_signed_rad_annotation_conversion_preserves_axis_conventions():
    import numpy as np
    from radsimreal.datasets.bev_dataset import BEVDataset
    from radsimreal.native.profiles import NativeProfile
    profile = NativeProfile("raddet",shape_native=(32,16,8),representation="complex",
                            range_reversed=True,azimuth_reversed=True,doppler_positive="approaching")
    # Native center is R=5,A=3,D=2 (absolute index, as in released RADDet gt) before its reversal.
    converted = BEVDataset._canonical_rad_boxes([[5,3,2,4,2,2]],profile)
    np.testing.assert_allclose(converted,[[26/32,12/16,5/8,4/32,2/16,2/8]])


def test_rad_pretraining_resume_preserves_history_optimizer_and_historical_best(tmp_path,monkeypatch):
    from radsimreal.bev.rad_pretraining import train_rad_backbone
    import radsimreal.bev.rad_pretraining as pretraining
    from radsimreal.config import RadarConfig,SimulationConfig
    class TinyRAD:
        def __init__(self,split):
            self.split = split
            self.samples = [{"id":split+"_0"}]
            self.normalization = {"fixed":True}
            self.config = SimulationConfig(radar=RadarConfig(range_bins=16,azimuth_bins=16,doppler_bins=4,range_step=1))
            self.radar_shape = (4,16,16)
        def __len__(self):
            return 1
        def __getitem__(self,index):
            generator = torch.Generator().manual_seed(4 if self.split == "train" else 9)
            return {"radar":torch.randn(4,16,16,generator=generator),
                    "boxes_rad":torch.tensor([[.5,.5,.5,.1,.2,.3]])}
        def get_meta(self):
            return {"radar_config":self.config.to_dict()["radar"]}
    train,val = TinyRAD("train"),TinyRAD("val")
    monkeypatch.setattr(pretraining,"_validation_loss",lambda *args:.1)
    first,_ = train_rad_backbone(train,val,tmp_path/"first",epochs=1,max_steps=1,batch_size=1,learning_rate=.007)
    original = first.read_bytes()
    monkeypatch.setattr(pretraining,"_validation_loss",lambda *args:.9)
    best,history = train_rad_backbone(train,val,tmp_path/"resumed",epochs=1,max_steps=2,batch_size=1,
                                    learning_rate=.3,resume=first.parent/"last.pt")
    last = torch.load(best.parent/"last.pt",weights_only=True)
    assert last["global_step"] == 2
    assert [v["epoch"] for v in history] == [0,1]
    assert last["best_val_loss"] == .1
    assert best.read_bytes() == original
    assert last["training_recipe"]["effective_parameter_groups"][0]["learning_rate"] == .007
    assert last["resume_sensor_comparison"]["sampling_equal"] is True
    assert last["best_selection_scope"] == "continued_run_with_historical_best_weights"
    train.config.radar.range_step = val.config.radar.range_step = 2
    with pytest.raises(ValueError,match="sensor sampling"):
        train_rad_backbone(train,val,tmp_path/"different_sensor",epochs=1,max_steps=3,resume=best.parent/"last.pt")
    train.config.radar.range_step = val.config.radar.range_step = 1
    val.samples[0]["id"] = train.samples[0]["id"]
    with pytest.raises(ValueError,match="IDs overlap"):
        train_rad_backbone(train,val,tmp_path/"leak",epochs=1,max_steps=1)


def test_checkpoint_write_failure_preserves_previous_weights(tmp_path,monkeypatch):
    from pathlib import Path
    from radsimreal.bev.training import _save_checkpoint
    path = tmp_path/"last.pt"
    path.write_bytes(b"previous_complete_checkpoint")
    def interrupted(payload,temporary):
        Path(temporary).write_bytes(b"partial_new_checkpoint")
        raise OSError("simulated interrupted write")
    monkeypatch.setattr(torch,"save",interrupted)
    with pytest.raises(OSError,match="interrupted"):
        _save_checkpoint({"some":"state"},path)
    assert path.read_bytes() == b"previous_complete_checkpoint"
    assert list(tmp_path.glob("*.tmp-*")) == []
