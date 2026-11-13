"""Physical geometry and actual Cartesian detector execution checks."""
import math
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from radsimreal.bev import (BEVGrid,PolarToBEV,create_model,bev_detection_loss,
    detect_bev,rotated_iou,rotated_nms,oriented_corners,train_bev,load_bev_checkpoint,
    evaluate_bev,evaluate_predictions_bev)
from radsimreal.bev.evaluation import rotated_average_precision
from radsimreal.bev.losses import anchor_targets


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_cartesian_grid_roundtrip_and_clipping():
    grid = BEVGrid(x_min=-8,x_max=8,y_min=0,y_max=16,height=32,width=32)
    boxes = np.array([[-2,3,4,7],[-10,-1,3,5]],dtype=float)
    np.testing.assert_allclose(grid.denormalize_boxes(grid.normalize_boxes(boxes)),boxes)
    clipped,valid = grid.clip_boxes(boxes)
    np.testing.assert_allclose(clipped[1],[-8,0,3,5])
    assert valid.tolist() == [True,True]
    x,y = grid.index_to_coord(3,5)
    np.testing.assert_allclose(grid.coord_to_index(x,y),[3,5])
    with pytest.raises(ValueError):
        BEVGrid(x_min=1,x_max=0)


def test_projection_matches_independent_analytic_linear_field_and_gradients():
    ranges = np.array([0,.5,1.5,3,5,8])
    angular = np.array([-1,-.3,0,.2,1])
    grid = BEVGrid(x_min=-2,x_max=2,y_min=1,y_max=5,height=4,width=4)
    projector = PolarToBEV(ranges,angular,grid,"sin")
    image = torch.tensor(ranges[:,None]+3*angular[None],dtype=torch.float64,requires_grad=True)
    projected = projector(image)
    xx,yy = np.meshgrid(grid.x_centers,grid.y_centers)
    expected = np.hypot(xx,yy)+3*xx/np.hypot(xx,yy)
    np.testing.assert_allclose(projected.detach(),expected,rtol=1e-6,atol=1e-6)
    projected.sum().backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0


def test_projection_masks_outside_fov_and_behind_sensor():
    grid = BEVGrid(x_min=-4,x_max=4,y_min=-2,y_max=2,height=4,width=8)
    projector = PolarToBEV([0,1,2],[-.5,0,.5],grid)
    result = projector(torch.ones(3,3))
    assert torch.equal(result[:2],torch.zeros_like(result[:2]))
    assert torch.equal(result[~projector.valid_mask],torch.zeros_like(result[~projector.valid_mask]))
    with pytest.raises(ValueError):
        PolarToBEV([0,1,1],[-1,0,1],grid)
    with pytest.raises(ValueError,match="complex"):
        projector(torch.ones(3,3,dtype=torch.complex64))
    with pytest.raises(ValueError,match="finite"):
        projector(torch.full((3,3),float("nan")))


def test_rotated_polygon_iou_known_values_and_convention():
    box = [0,0,2,4,0]
    comparison = [[0,0,2,4,0],[0,0,2,4,math.pi/2],[20,0,2,4,0]]
    np.testing.assert_allclose(rotated_iou([box],comparison),[[1,1/3,0]],atol=1e-12)
    square = [0,0,2,2,0]
    assert rotated_iou([square],[[0,0,2,2,math.pi/4]])[0,0] == pytest.approx(1/math.sqrt(2))
    corners = oriented_corners([0,0,2,4,math.pi/2])
    np.testing.assert_allclose(corners.max(0)-corners.min(0),[4,2],atol=1e-12)
    assert rotated_iou([],comparison).shape == (0,3)
    with pytest.raises(ValueError):
        rotated_iou([[0,0,-1,2,0]],[box])


def test_rotated_nms_and_ap_duplicate_matching():
    box = [0,0,2,4,0]
    boxes = torch.tensor([box,box,[20,0,2,4,0]],dtype=torch.float32)
    assert rotated_nms(boxes,torch.tensor([.9,.8,.7]),.1).tolist() == [0,2]
    predictions = [{"oriented_boxes":boxes,"scores":[.9,.8,.7]}]
    metrics = rotated_average_precision(predictions,[boxes[[0,2]]],.5)
    assert metrics["ap"] == pytest.approx(5/6)
    assert metrics["false_positives"] == 1
    assert rotated_average_precision([{"oriented_boxes":[],"scores":[]}],[[]])["ap"] is None


@pytest.mark.parametrize("name",["unet","raddet","probabilistic"])
def test_all_model_families_forward_backward_and_physical_decode(name):
    torch.manual_seed(41)
    grid = BEVGrid(x_min=-8,x_max=8,y_min=0,y_max=16,height=32,width=32)
    model = create_model(name,grid=grid,base_channels=4,input_shape=(4,32,32))
    image = torch.randn(2,3,32,32)
    radar = torch.randn(2,4,32,32)
    boxes = [torch.tensor([[.2,.2,.4,.5]]),torch.empty(0,4)]
    output = model(image,radar)
    loss = bev_detection_loss(output,boxes,grid=grid)
    assert torch.isfinite(loss["total"])
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    parameter = next(model.parameters())
    previous = parameter.detach().clone()
    loss["total"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(parameter,previous)
    results = detect_bev(model,image,radar,score_threshold=0,top_k=3)
    assert len(results) == 2
    assert all(p["box_domain"] == "cartesian_bev_metres_axis_aligned" for p in results)
    assert all(len(p["boxes"]) <= 3 for p in results)
    assert all(torch.isfinite(p["boxes"]).all() for p in results)
    assert model.training


def test_probabilistic_oriented_labels_are_required_and_variance_is_trained():
    grid = BEVGrid(x_min=-8,x_max=8,y_min=0,y_max=16,height=32,width=32)
    model = create_model("probabilistic",grid=grid,base_channels=4,oriented=True)
    output = model(torch.randn(1,3,32,32))
    boxes = [torch.tensor([[.4,.4,.6,.7]])]
    with pytest.raises(ValueError,match="genuine"):
        bev_detection_loss(output,boxes,grid=grid)
    oriented = [torch.tensor([[0,8,2,4,.4]])]
    losses = bev_detection_loss(output,boxes,oriented,grid)
    losses["total"].backward()
    assert torch.isfinite(model.head.weight.grad).all()
    result = detect_bev(model,torch.zeros(1,3,32,32),score_threshold=0,top_k=2)[0]
    assert result["oriented_boxes"].shape[1] == 5
    assert result["box_domain"] == "cartesian_bev_metres_oriented"


class FixtureBEVDataset(Dataset):
    def __init__(self,split="train",normalization=None):
        self.grid = BEVGrid(x_min=-8,x_max=8,y_min=0,y_max=16,height=32,width=32)
        self.normalization = {"fixed_fixture":True} if normalization is None else normalization
        self.label_domain = "fixture_true_cartesian_footprints"
        self.radar_shape = (4,32,32)
        from radsimreal.config import RadarConfig,SimulationConfig
        self.config = SimulationConfig(radar=RadarConfig(range_bins=32,azimuth_bins=32,doppler_bins=4,range_step=.5))
        self.samples = [{"id":f"{split}_{i}"} for i in range(2)]
        self.split = split
    def __len__(self):
        return 2
    def get_meta(self):
        return {"source":"synthetic_fixture_only","split":self.split,"radar_config":self.config.to_dict()["radar"]}
    def __getitem__(self,i):
        generator = torch.Generator().manual_seed(i+23)
        boxes = torch.tensor([[.2,.3,.4,.6]]) if i == 0 else torch.empty(0,4)
        return {"image":torch.randn(3,32,32,generator=generator),"radar":torch.randn(*self.radar_shape,generator=generator),
                "boxes":boxes,"boxes_physical":self.grid.denormalize_boxes(boxes),"oriented_boxes":torch.empty(0,5),
                "id":self.samples[i]["id"]}


@pytest.mark.parametrize("name",["unet","raddet","probabilistic"])
def test_all_models_train_validation_checkpoint_evaluate_and_resume(tmp_path,name):
    train,val = FixtureBEVDataset(),FixtureBEVDataset("val")
    checkpoint,history = train_bev(train,val,tmp_path/name,model_name=name,epochs=1,batch_size=2,max_steps=1,base_channels=4)
    model,payload = load_bev_checkpoint(checkpoint)
    assert payload["global_step"] == 1
    assert payload["metric_domain"] == "cartesian_bev_metres_axis_aligned"
    assert math.isfinite(history[0]["val_loss"])
    result = evaluate_bev(checkpoint,FixtureBEVDataset("test"),top_k=2)
    assert result["num_images"] == 2
    assert result["metric_domain"] == "cartesian_bev_metres_axis_aligned"
    assert result["label_domain"] == train.label_domain
    assert all(0 <= result[k] <= 1 for k in ("AP@0.1","AP@0.3","AP@0.5"))
    resumed,_ = train_bev(train,val,tmp_path/f"{name}_resume",epochs=1,max_steps=2,batch_size=2,resume=checkpoint)
    _,last = load_bev_checkpoint(resumed.parent/"last.pt")
    assert last["global_step"] == 2


def test_evaluator_domain_guard_and_unclipped_ground_truth(tmp_path):
    train,val = FixtureBEVDataset(),FixtureBEVDataset("val")
    checkpoint,_ = train_bev(train,val,tmp_path/"model",epochs=1,max_steps=1,base_channels=2)
    test = FixtureBEVDataset("test")
    test.label_domain = "different_annotation_definition"
    with pytest.raises(ValueError,match="label domain"):
        evaluate_bev(checkpoint,test,top_k=1)
    result = evaluate_bev(checkpoint,test,top_k=1,allow_label_domain_change=True)
    assert result["cross_label_domain_comparison"]
    with pytest.raises(ValueError,match="non-Cartesian"):
        evaluate_predictions_bev([{"boxes":[],"scores":[],"box_domain":"normalized_ra"}],[[]])
    # IoU is1/3 on physical ground plane; clipping target to prediction would give1.
    metrics = evaluate_predictions_bev([{"boxes":[[0,0,1,1]],"scores":[1]}],[[[-1,0,2,1]]],iou_thresholds=(.5,))
    assert metrics["AP@0.5"] == 0


def test_raddet_original_rad_pretrain_then_frozen_cartesian_stage(tmp_path):
    from radsimreal.bev.rad_pretraining import train_rad_backbone,RADDetRAD,rad_loss,decode_rad,iou_rad
    class RADFixture(FixtureBEVDataset):
        def __getitem__(self,i):
            sample = super().__getitem__(i)
            sample["boxes_rad"] = torch.tensor([[.3,.4,.5,.15,.2,.25]]) if i == 0 else torch.empty(0,6)
            return sample
    train,val = RADFixture(),RADFixture("val")
    pretrain,history = train_rad_backbone(train,val,tmp_path/"rad",epochs=1,max_steps=1,batch_size=2)
    assert math.isfinite(history[0]["val_loss"])
    source = torch.load(pretrain,weights_only=True)
    checkpoint,_ = train_bev(train,val,tmp_path/"cart",model_name="raddet",epochs=1,max_steps=1,batch_size=2,
                             pretrained_backbone=pretrain,freeze_backbone=True)
    model,payload = load_bev_checkpoint(checkpoint)
    assert payload["training_recipe"]["backbone_frozen"]
    assert payload["backbone_transfer"]["global_step"] == 1
    for key,value in source["model_state"].items():
        if key.startswith("backbone."):
            assert torch.equal(value,payload["model_state"][key])
    resumed,_ = train_bev(train,val,tmp_path/"resumed_cart",epochs=1,max_steps=2,batch_size=2,resume=checkpoint)
    _,last = load_bev_checkpoint(resumed.parent/"last.pt")
    assert last["training_recipe"]["backbone_frozen"]
    for key,value in source["model_state"].items():
        if key.startswith("backbone."):
            assert torch.equal(value,last["model_state"][key])
    box = torch.tensor([[.3,.4,.5,.15,.2,.25]])
    assert iou_rad(box,box).item() == pytest.approx(1)


def test_literal_raddet_decoder_clamp_option_is_explicit():
    from radsimreal.bev.losses import decode_anchor_parameters
    raw = torch.zeros(1,1,1,1,6)
    raw[...,2:4] = math.log(.5)
    output = {"head_type":"raddet_anchor","raw":raw,"anchors":torch.tensor([[.2,.4]])}
    assert torch.allclose(decode_anchor_parameters(output)[1].reshape(2),torch.tensor([.1,.2]))
    output["upstream_size_clamp"] = True
    assert torch.allclose(decode_anchor_parameters(output)[1].reshape(2),torch.tensor([.2,.4]))


def test_probabilistic_inference_exposes_physical_uncertainty_and_masks_untrained_yaw():
    from radsimreal.bev.inference import decode_bev
    grid = BEVGrid(x_min=-10,x_max=10,y_min=0,y_max=40,height=4,width=4)
    raw = torch.zeros(1,1,1,1,13)
    raw[...,0] = 10
    raw[...,3] = math.log(2)
    raw[...,4] = math.log(.5)
    raw[...,5] = 1
    raw[...,7:13] = torch.log(torch.tensor([.5,.25,.25,.5,.3,.4]))
    output = {"head_type":"probabilistic_anchor","raw":raw,"anchors":torch.tensor([[.1,.2]]),
              "anchor_angles":torch.tensor([0.]),"oriented":True}
    prediction = decode_bev(output,grid,score_threshold=.9)[0]
    assert prediction["localization_variance"].shape == (1,6)
    np.testing.assert_allclose(prediction["box_parameter_variance_m2"],[[2,16,4,8]],rtol=1e-6)
    np.testing.assert_allclose(prediction["unclipped_box_parameters_m"],[[0,20,4,4]],atol=1e-6)
    assert prediction["orientation_variance_rad2"] == pytest.approx([.4])
    assert prediction["orientation_uncertainty_valid"].tolist() == [True]
    assert not prediction["uncertainty_calibrated_on_real_data"]
    output["oriented"] = False
    prediction = decode_bev(output,grid,score_threshold=.9)[0]
    assert prediction["orientation_variance_rad2"] == [None]
    assert prediction["localization_variance_supervision_mask"] == [True,True,True,True,False,False]
    output["oriented"] = True
    raw[...,5:7] = 0
    prediction = decode_bev(output,grid,score_threshold=.9)[0]
    assert prediction["orientation_variance_rad2"] == [None]
    assert prediction["orientation_uncertainty_valid"].tolist() == [False]
    empty = decode_bev(output,grid,score_threshold=1)[0]
    assert empty["localization_variance"].shape == (0,6)
    assert empty["orientation_variance_rad2"] == []


@pytest.mark.parametrize("name",["unet","raddet"])
def test_sensor_sampling_changes_are_reported_and_native_models_require_opt_in(tmp_path,name):
    train,val = FixtureBEVDataset(),FixtureBEVDataset("val")
    checkpoint,_ = train_bev(train,val,tmp_path/name,model_name=name,epochs=1,max_steps=1,batch_size=2,base_channels=2)
    test = FixtureBEVDataset("test")
    test.config.radar.range_step *= 2
    if name == "raddet":
        with pytest.raises(ValueError,match="sensor sampling"):
            evaluate_bev(checkpoint,test,top_k=1)
    result = evaluate_bev(checkpoint,test,top_k=1,allow_sensor_change=name == "raddet")
    comparison = result["sensor_sampling_comparison"]
    assert comparison["sampling_equal"] is False
    assert comparison["changed_fields"]["range_step"] == {"training":.5,"current":1.}
    assert result["sensor_change_explicitly_allowed"] == (name == "raddet")
    if name == "raddet":
        test.radar_shape = (4,64,32)
        with pytest.raises(ValueError,match="fixed architecture"):
            evaluate_bev(checkpoint,test,top_k=1,allow_sensor_change=True)


def test_sensor_comparison_ignores_noise_but_rejects_unproven_raw_geometry(tmp_path):
    from radsimreal.bev.training import compare_sensor_sampling
    train,val = FixtureBEVDataset(),FixtureBEVDataset("val")
    config = train.get_meta()
    val.config.radar.noise_std = 3
    val.config.radar.scattering_gain = 100
    val.config.radar.psf_path = "unrelated_calibration_path.npz"
    assert compare_sensor_sampling(config,val)["sampling_equal"] is True
    checkpoint,_ = train_bev(train,val,tmp_path/"raddet",model_name="raddet",epochs=1,max_steps=1,batch_size=2)
    unknown = FixtureBEVDataset("test")
    unknown.config = None
    unknown.get_meta = lambda:{"source":"missing_sampling_provenance"}
    with pytest.raises(ValueError,match="lacks provenance"):
        evaluate_bev(checkpoint,unknown,top_k=1)
    result = evaluate_bev(checkpoint,unknown,top_k=1,allow_sensor_change=True)
    assert result["sensor_sampling_comparison"]["sampling_equal"] is None


def test_ap_matching_policy_matches_released_raddet_duplicate_rule():
    predictions = [{"boxes":[[0,0,1,1],[0,0,1,1]],"scores":[.9,.8]}]
    targets = [[[0,0,1,1],[.5,0,1.5,1]]]
    # Second detection's highest overlap is the already matched first GT.
    reference = evaluate_predictions_bev(predictions,targets,iou_thresholds=(.3,),matching_policy="best_overall")
    alternate = evaluate_predictions_bev(predictions,targets,iou_thresholds=(.3,),matching_policy="best_unmatched")
    assert reference["AP@0.3"] == pytest.approx(.5)
    assert alternate["AP@0.3"] == pytest.approx(1.)
    assert reference["matching_policy"] == "best_overall"
    oriented_predictions = [{"oriented_boxes":[[.5,.5,1,1,0],[.5,.5,1,1,0]],"scores":[.9,.8]}]
    oriented_targets = [[[.5,.5,1,1,0],[1,.5,1,1,0]]]
    assert rotated_average_precision(oriented_predictions,oriented_targets,.3,"best_overall")["ap"] == pytest.approx(.5)
    assert rotated_average_precision(oriented_predictions,oriented_targets,.3,"best_unmatched")["ap"] == pytest.approx(1.)


def test_reference_frame_mean_and_global_ap_have_explicit_empty_frame_semantics():
    predictions = [{"boxes":[[0,0,1,1]],"scores":[.5]},
                   {"boxes":[[3,3,4,4]],"scores":[.9]}]
    targets = [[[0,0,1,1]],[]]
    global_result = evaluate_predictions_bev(predictions,targets,iou_thresholds=(.5,),ap_aggregation="global_pr")
    frame_result = evaluate_predictions_bev(predictions,targets,iou_thresholds=(.5,),ap_aggregation="frame_mean")
    assert global_result["AP@0.5"] == pytest.approx(.5)
    assert frame_result["AP@0.5"] == pytest.approx(1.)
    assert frame_result["curves"]["0.5"]["num_gt_empty_frames_excluded"] == 1
    assert frame_result["curves"]["0.5"]["num_predictions_ignored_in_gt_empty_frames"] == 1
    assert frame_result["curves"]["0.5"]["per_frame_ap"] == [1.,None]
    assert evaluate_predictions_bev(predictions,[[],[]],iou_thresholds=(.5,),ap_aggregation="global_pr")["AP@0.5"] is None
    assert evaluate_predictions_bev(predictions,[[],[]],iou_thresholds=(.5,),ap_aggregation="frame_mean")["AP@0.5"] == 0


def test_raddet_objectness_score_and_physical_prediction_extents_follow_source():
    from radsimreal.bev.inference import decode_bev
    from radsimreal.models.inference import decode_detections
    grid = BEVGrid(x_min=-10,x_max=10,y_min=0,y_max=40,height=4,width=4)
    raw = torch.zeros(1,1,1,1,6)
    raw[...,0] = torch.logit(torch.tensor(.01))
    raw[...,5] = -100  # Class logit must not alter reference objectness ranking.
    output = {"head_type":"raddet_anchor","raw":raw,"anchors":torch.tensor([[.2,.2]])}
    prediction = decode_bev(output,grid,score_threshold=.4)[0]
    assert prediction["scores"].tolist() == [.5]
    np.testing.assert_allclose(prediction["boxes"],[[-11.8,16,-7.8,24]],atol=1e-5)
    assert prediction["prediction_clipping"] == "none"
    assert prediction["score_definition"] == "sigmoid of objectness logit"
    center = {"head_type":"center","heatmap":torch.zeros(1,1,1,1),
              "offset":torch.tensor([[[[.01]],[[.5]]]]),
              "size":torch.full((1,2,1,1),.2)}
    physical = decode_bev(center,grid,score_threshold=.4)[0]
    np.testing.assert_allclose(physical["boxes"],[[-11.8,16,-7.8,24]],atol=1e-5)
    # Preserve the public v1 RA decoder's clipping default.
    legacy = decode_detections(center,score_threshold=.4)[0]
    assert legacy["boxes"][0,0] == 0
