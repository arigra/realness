"""Training/checkpoint integration for all Cartesian detector families."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import random
import shutil
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader
from .geometry import BEVGrid
from .models import create_model
from .losses import bev_detection_loss


def collate_bev(items):
    batch = {"image":torch.stack([torch.as_tensor(v["image"]).float() for v in items]),
             "boxes":[torch.as_tensor(v["boxes"]).float().reshape(-1,4) for v in items],
             "oriented_boxes":[torch.as_tensor(v.get("oriented_boxes",[])).float().reshape(-1,5) for v in items],
             "id":[str(v.get("id",i)) for i,v in enumerate(items)]}
    if all("radar" in v for v in items):
        batch["radar"] = torch.stack([torch.as_tensor(v["radar"]).float() for v in items])
    elif any("radar" in v for v in items):
        raise ValueError("Either every sample in a batch supplies radar or none does")
    for key,dimension in (("boxes_physical",4),("oriented_boxes_physical",5)):
        if all(key in v for v in items):
            batch[key] = [torch.as_tensor(v[key]).float().reshape(-1,dimension) for v in items]
        elif any(key in v for v in items):
            raise ValueError(f"Every sample must consistently provide {key}")
    return batch


def _save_checkpoint(payload,path):
    """Replace only after a complete write, retaining prior checkpoints on error."""
    path = Path(path)
    temporary = path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    try:
        torch.save(payload,temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value,dict):
        return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_plain(v) for v in value]
    if isinstance(value,(np.ndarray,np.generic)):
        return _plain(value.tolist())
    if isinstance(value,Path):
        return str(value)
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    raise TypeError(f"Unsupported checkpoint metadata type {type(value).__name__}")


_SENSOR_KEYS = ("range_bins","doppler_bins","azimuth_bins","range_min","range_step",
                "velocity_min","velocity_step","azimuth_coordinate","azimuth_min_deg","azimuth_max_deg")


def _dataset_radar_config(dataset):
    config = getattr(dataset,"config",None)
    radar = getattr(config,"radar",None)
    if radar is not None:
        if is_dataclass(radar):
            return _plain(asdict(radar))
        if isinstance(radar,dict):
            return _plain(radar)
    metadata = getattr(dataset,"get_meta",lambda:{})()
    value = metadata.get("radar_config") if isinstance(metadata,dict) else None
    return _plain(value) if isinstance(value,dict) else None


def compare_sensor_sampling(reference_meta,dataset):
    """Compare actual tensor coordinates, excluding noise/material/path choices.

    Cartesian reprojection allows a BEV-input detector to receive a different
    polar sampling, but this is still a reported sensor-domain change. A RADDet
    native-tensor model depends directly on this sampling and requires an
    explicit opt-in. Missing provenance cannot establish equality.
    """
    reference = reference_meta.get("radar_config") if isinstance(reference_meta,dict) else None
    current = _dataset_radar_config(dataset)
    if not isinstance(reference,dict) or not isinstance(current,dict):
        return {"status":"unknown_missing_sampling_provenance","sampling_equal":None,
                "changed_fields":{},"missing_reference_fields":list(_SENSOR_KEYS) if not isinstance(reference,dict) else [k for k in _SENSOR_KEYS if k not in reference],
                "missing_current_fields":list(_SENSOR_KEYS) if not isinstance(current,dict) else [k for k in _SENSOR_KEYS if k not in current],
                "reference":{k:reference[k] for k in _SENSOR_KEYS if isinstance(reference,dict) and k in reference},
                "current":{k:current[k] for k in _SENSOR_KEYS if isinstance(current,dict) and k in current}}
    missing_ref = [k for k in _SENSOR_KEYS if k not in reference]
    missing_current = [k for k in _SENSOR_KEYS if k not in current]
    changed = {}
    for key in _SENSOR_KEYS:
        if key not in reference or key not in current:
            continue
        before,after = reference[key],current[key]
        if isinstance(before,str) or isinstance(after,str):
            same = before == after
        else:
            try:
                same = bool(np.isclose(float(before),float(after),rtol=1e-9,atol=1e-10))
            except (ValueError,TypeError):
                same = False
        if not same:
            changed[key] = {"training":before,"current":after}
    known = not missing_ref and not missing_current
    equal = not changed if known else None
    return {"status":"same_declared_sampling" if equal else "different_declared_sampling" if changed else "unknown_missing_sampling_provenance",
            "sampling_equal":equal,"changed_fields":changed,"missing_reference_fields":missing_ref,
            "missing_current_fields":missing_current,
            "reference":{k:reference[k] for k in _SENSOR_KEYS if k in reference},
            "current":{k:current[k] for k in _SENSOR_KEYS if k in current}}


def enforce_sensor_sampling(model,comparison,dataset,allow_sensor_change=False):
    if model.requires_radar:
        shape = getattr(dataset,"radar_shape",None)
        if shape is not None and tuple(shape) != tuple(model.input_shape):
            raise ValueError("RADDet native input shape differs from its fixed architecture; a sensor-change opt-in cannot resize the model")
        if comparison["sampling_equal"] is not True and not allow_sensor_change:
            raise ValueError("RADDet native tensor sensor sampling differs or lacks provenance; set allow_sensor_change=True for an explicit sensor-domain transfer")


def load_bev_checkpoint(path, device="cpu"):
    payload = torch.load(Path(path),map_location=device,weights_only=True)
    if payload.get("format_version") != "bev_v2" or not isinstance(payload.get("model_config"),dict):
        raise ValueError("Checkpoint is not a Cartesian BEV v2 model")
    model = create_model(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model,payload


def _forward_loss(model,batch,device):
    images = batch["image"].to(device)
    radar = batch.get("radar")
    if model.requires_radar and radar is None:
        raise ValueError("RADDet training requires the native normalized full Doppler tensor")
    radar = radar.to(device) if model.requires_radar else None
    targets = [v.to(device) for v in batch["boxes"]]
    oriented = [v.to(device) for v in batch["oriented_boxes"]]
    return bev_detection_loss(model(images,radar=radar),targets,oriented,model.grid)


def _validate(model,loader,device):
    model.eval()
    total,n = 0.,0
    with torch.no_grad():
        for batch in loader:
            loss = _forward_loss(model,batch,device)["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite BEV validation loss")
            total += float(loss)*len(batch["image"])
            n += len(batch["image"])
    return total/n


def _matching_prior_best(resume,payload):
    candidates = [Path(resume).parent/"best.pt",Path(resume)]
    expected = payload.get("best_val_loss")
    history = payload.get("history",[])
    for path in candidates:
        if not path.is_file():
            continue
        other = torch.load(path,map_location="cpu",weights_only=True)
        if other.get("format_version") != "bev_v2" or other.get("model_config") != payload.get("model_config"):
            continue
        other_history = other.get("history",[])
        if not other_history or other_history != history[:len(other_history)]:
            continue
        if other_history[-1].get("val_loss") == expected:
            return path
    return None


def train_bev(train_dataset,val_dataset,output_dir,model_name="unet",epochs=10,batch_size=4,
              max_steps=None,base_channels=16,device="cpu",seed=2024,learning_rate=None,
              weight_decay=None,optimizer_name=None,resume=None,num_threads=1,
              oriented=False,anchors=None,model_options=None,pretrained_backbone=None,
              freeze_backbone=False,allow_sensor_change=False,num_workers=0):
    """Train on Cartesian labels, preserving dataset/grid/metric provenance.

    `boxes` are normalized Cartesian XYXY. `oriented_boxes` are true physical
    [cx,cy,width,height,theta_ccw] where available. No orientation is invented.
    Resume restarts at the next epoch and reseeds; interrupted shuffle/RNG state
    is not a bitwise continuation. max_steps is total optimizer steps.
    """
    model_name = {"coordinate_unet_bev":"unet","zhang_presence_unet_bev":"unet_zhang","raddet_cartesian_port":"raddet",
                  "dong_probabilistic_bev":"probabilistic"}.get(model_name,model_name)
    if epochs < 1 or batch_size < 1 or num_threads < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("epochs,batch_size,num_threads,max_steps must be positive")
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Nonempty distinct training and validation datasets are required")
    if train_dataset is val_dataset:
        raise ValueError("Training and validation cannot be the same dataset instance")
    if getattr(train_dataset,"split","train") != "train" or getattr(val_dataset,"split","val") == "test":
        raise ValueError("Training/model selection must not use the held-out test split")
    if train_dataset.grid.to_dict() != val_dataset.grid.to_dict():
        raise ValueError("Train/validation BEV grids differ")
    label_domain = str(getattr(train_dataset,"label_domain","unspecified_cartesian_labels"))
    if label_domain != str(getattr(val_dataset,"label_domain",label_domain)):
        raise ValueError("Train/validation label domains differ")
    normalization = _plain(dict(getattr(train_dataset,"normalization",{})))
    if normalization != _plain(dict(getattr(val_dataset,"normalization",{}))):
        raise ValueError("Train/validation preprocessing differs")
    train_ids = {str(v["id"]) for v in getattr(train_dataset,"samples",[]) if isinstance(v,dict) and "id" in v}
    val_ids = {str(v["id"]) for v in getattr(val_dataset,"samples",[]) if isinstance(v,dict) and "id" in v}
    if train_ids & val_ids:
        raise ValueError("Train and validation IDs overlap")
    torch.set_num_threads(num_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    best_path = output_dir/"best.pt"
    checkpoint = None
    backbone_transfer = None
    if resume is not None:
        if pretrained_backbone is not None:
            raise ValueError("Resume already supplies model weights; do not also request backbone initialization")
        model,checkpoint = load_bev_checkpoint(resume,device)
        if checkpoint["normalization"] != normalization or checkpoint["grid"] != train_dataset.grid.to_dict() or checkpoint["label_domain"] != label_domain:
            raise ValueError("Resume dataset grid, normalization or annotation domain differs")
        model_name = checkpoint["model_config"]["name"]
        freeze_backbone = bool(checkpoint.get("training_recipe",{}).get("backbone_frozen",False))
        if freeze_backbone:
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(False)
        backbone_transfer = checkpoint.get("backbone_transfer")
    else:
        config = dict(model_options or {})
        config.update(grid=train_dataset.grid,base_channels=base_channels,oriented=oriented)
        if anchors is not None:
            config["anchors"] = anchors
        if model_name == "raddet":
            shape = getattr(train_dataset,"radar_shape",None)
            if shape is None:
                sample = train_dataset[0]
                if "radar" not in sample:
                    raise ValueError("RADDet requires full native tensor input")
                shape = sample["radar"].shape
            config["input_shape"] = tuple(shape)
        model = create_model(model_name,**config).to(device)
        if pretrained_backbone is not None:
            from .rad_pretraining import load_pretrained_raddet_backbone
            backbone_transfer = load_pretrained_raddet_backbone(model,pretrained_backbone,freeze_backbone,normalization,
                dataset_meta=getattr(train_dataset,"get_meta",lambda:{})())
        elif freeze_backbone:
            raise ValueError("Freezing an uninitialized backbone requires a pretrained_backbone checkpoint")
    dataset_meta = _plain(getattr(train_dataset,"get_meta",lambda:{})())
    actual_radar_config = _dataset_radar_config(train_dataset)
    if actual_radar_config is not None:
        dataset_meta["radar_config"] = actual_radar_config
    validation_sensor = compare_sensor_sampling(dataset_meta,val_dataset)
    enforce_sensor_sampling(model,validation_sensor,val_dataset,allow_sensor_change)
    resume_sensor = None
    if checkpoint is not None:
        resume_sensor = compare_sensor_sampling(checkpoint.get("dataset_meta",{}),train_dataset)
        enforce_sensor_sampling(model,resume_sensor,train_dataset,allow_sensor_change)
    learning_rate = (1e-4 if model_name == "raddet" else 5e-4 if model_name == "probabilistic" else 1e-3) if learning_rate is None else learning_rate
    weight_decay = (1e-4 if model_name in ("unet","unet_zhang") else 0.) if weight_decay is None else weight_decay
    optimizer_name = ("AdamW" if model_name in ("unet","unet_zhang") else "Adam") if optimizer_name is None else optimizer_name
    if checkpoint is not None:
        optimizer_name = checkpoint["training_recipe"]["optimizer"]
    if learning_rate <= 0 or weight_decay < 0 or optimizer_name.lower() not in ("adam","adamw"):
        raise ValueError("Invalid optimizer settings")
    optimizer_class = torch.optim.AdamW if optimizer_name.lower() == "adamw" else torch.optim.Adam
    optimizer = optimizer_class([p for p in model.parameters() if p.requires_grad],lr=learning_rate,weight_decay=weight_decay)
    start_epoch,global_step,best_val,history = 0,0,float("inf"),[]
    selection_scope = "complete_run"
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch,global_step = checkpoint["epoch"]+1,checkpoint["global_step"]
        history = list(checkpoint["history"])
        prior_best = _matching_prior_best(resume,checkpoint)
        if prior_best is not None:
            best_val = checkpoint["best_val_loss"]
            if prior_best.resolve() != best_path.resolve():
                shutil.copy2(prior_best,best_path)
            selection_scope = "continued_run_with_historical_best_weights"
        else:
            selection_scope = "resumed_segment_only_historical_best_weights_unavailable"
    if max_steps is not None and global_step >= max_steps:
        raise ValueError("max_steps must exceed resumed update count")
    # Dong2020 uses cosine periods of100k updates. No scheduler was disclosed
    # for RadSimReal's adaptation; retain this named reference recipe explicitly.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100000) if model_name == "probabilistic" else None
    if scheduler is not None and checkpoint and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    generator = torch.Generator().manual_seed(seed)
    workers = dict(num_workers=num_workers,persistent_workers=num_workers>0)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,generator=generator,collate_fn=collate_bev,**workers)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_bev,**workers)
    for epoch in range(start_epoch,start_epoch+epochs):
        model.train()
        if freeze_backbone:
            model.backbone.eval()
        sums = {"total":0.,"heatmap":0.,"box":0.,"collisions":0}
        n,updates = 0,0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses = _forward_loss(model,batch,device)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Nonfinite BEV training loss")
            losses["total"].backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for key in ("total","heatmap","box"):
                sums[key] += float(losses[key].detach())*len(batch["image"])
            sums["collisions"] += int(losses.get("collisions",0))
            n += len(batch["image"])
            updates += 1
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break
        val_loss = _validate(model,val_loader,device)
        history.append({"epoch":epoch,"global_step":global_step,"updates":updates,"train_loss":sums["total"]/n,
                        "train_heatmap_loss":sums["heatmap"]/n,"train_box_loss":sums["box"]/n,
                        "target_collisions":sums["collisions"],"val_loss":val_loss,"last_gradient_norm":float(gradient)})
        improved = val_loss < best_val
        best_val = min(best_val,val_loss)
        effective = [{"learning_rate":float(g["lr"]),"weight_decay":float(g["weight_decay"])} for g in optimizer.param_groups]
        payload = {"format_version":"bev_v2","model_config":model.export_config,"architecture":model.architecture,
                   "fidelity":model.fidelity,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
                   "scheduler_state":scheduler.state_dict() if scheduler else None,"grid":model.grid.to_dict(),
                   "label_domain":label_domain,"normalization":normalization,"dataset_meta":dataset_meta,
                   "metric_domain":"cartesian_bev_metres_oriented" if model.oriented else "cartesian_bev_metres_axis_aligned",
                   "epoch":epoch,"global_step":global_step,"best_val_loss":best_val,"history":history,
                   "best_selection_scope":selection_scope,"seed":seed,"published_training_completed":False,
                   "backbone_transfer":backbone_transfer,
                   "validation_sensor_comparison":validation_sensor,"resume_sensor_comparison":resume_sensor,
                   "allow_sensor_change":bool(allow_sensor_change),
                   "training_recipe":{"optimizer":optimizer_class.__name__,"effective_parameter_groups":effective,
                                      "scheduler":"cosine_Tmax100000" if scheduler else "constant",
                                      "gradient_clip_norm":10,"published_radsimreal_recipe":False}}
        payload["training_recipe"]["backbone_frozen"] = bool(freeze_backbone)
        _save_checkpoint(payload,output_dir/"last.pt")
        if improved:
            _save_checkpoint(payload,best_path)
        (output_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        if max_steps is not None and global_step >= max_steps:
            break
    return best_path,history
