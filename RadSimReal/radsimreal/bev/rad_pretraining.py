"""RADDet's original 3D RAD-head pretraining stage and frozen-backbone transfer.

This is a RAD-coordinate object task, distinct from Cartesian evaluation. Targets
are normalized (range,azimuth,Doppler center,width,height,depth), after the same
axis flips applied to the radar input. They are NOT Cartesian boxes. Native
annotation adapters must supply boxes_rad in exactly this declared convention.
"""
import json
import math
import shutil
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .models import RADDetBEV, conv_bn, _xavier


class RADDetRAD(nn.Module):
    requires_radar = True
    def __init__(self,input_shape=(64,256,256),anchors=None,repeats=(2,4,8,16),upstream_size_clamp=False):
        super().__init__()
        reference = RADDetBEV(input_shape=input_shape,repeats=repeats)
        self.backbone = reference.backbone
        self.input_shape = tuple(input_shape)
        self.repeats = tuple(repeats)
        d,r,a = self.input_shape
        self.rad_shape = (r,a,d)
        self.depth_cells = math.ceil(d/16)
        channels = d*4
        self.neck = conv_bn(channels,2*channels,keras=True)
        source = [(6,9,5),(19,36,60),(19,28,19),(19,17,8),(22,45,28),(26,75,61)]
        anchors = [[x/256,y/256,z/64] for x,y,z in source] if anchors is None else anchors
        anchors = torch.as_tensor(anchors,dtype=torch.float32)
        if anchors.ndim != 2 or anchors.shape[1] != 3 or not torch.isfinite(anchors).all() or (anchors <= 0).any():
            raise ValueError("RAD anchors require positive normalized R,A,D extents")
        self.register_buffer("anchors",anchors)
        self.head = nn.Conv2d(2*channels,self.depth_cells*len(anchors)*8,1)
        self.neck.apply(_xavier)
        self.head.apply(_xavier)
        self.upstream_size_clamp = bool(upstream_size_clamp)
        self.export_config = {"input_shape":list(input_shape),"anchors":anchors.tolist(),"repeats":list(repeats),
                              "upstream_size_clamp":self.upstream_size_clamp}

    def forward(self,radar):
        if tuple(radar.shape[1:]) != self.input_shape:
            raise ValueError("RAD pretraining input must be normalized B,D,R,A on the configured grid")
        features = self.neck(self.backbone(radar))
        b,_,r,a = features.shape
        raw = self.head(features).reshape(b,self.depth_cells,len(self.anchors),8,r,a).permute(0,4,5,1,2,3)
        return {"raw":raw,"anchors":self.anchors,"rad_shape":self.rad_shape,
                "upstream_size_clamp":self.upstream_size_clamp}


def decode_rad(outputs):
    raw = outputs["raw"]
    nr,na,nd = raw.shape[1:4]
    rr,aa,dd = torch.meshgrid(torch.arange(nr,device=raw.device,dtype=raw.dtype),
                               torch.arange(na,device=raw.device,dtype=raw.dtype),
                               torch.arange(nd,device=raw.device,dtype=raw.dtype),indexing="ij")
    coordinate = torch.stack([rr,aa,dd],-1)[None,:,:,:,None,:]
    center = (raw[...,:3].sigmoid()+coordinate)/raw.new_tensor([nr,na,nd])
    log_size = raw[...,3:6].clamp(1e-12,1e12) if outputs.get("upstream_size_clamp",False) else raw[...,3:6].clamp(-12,12)
    size = log_size.exp()*outputs["anchors"]
    # Released yoloheadToPredictions uses objectness rather than its product
    # with class probability. The class retains its own training loss.
    return torch.cat([center,size],-1),raw[...,6].sigmoid()


def iou_rad(boxes1,boxes2):
    """Pairwise axis-aligned3D overlap in declared normalized RAD coordinates."""
    a,b = boxes1.reshape(-1,6),boxes2.reshape(-1,6)
    lower = torch.maximum(a[:,None,:3]-a[:,None,3:]*.5,b[None,:,:3]-b[None,:,3:]*.5)
    upper = torch.minimum(a[:,None,:3]+a[:,None,3:]*.5,b[None,:,:3]+b[None,:,3:]*.5)
    intersection = (upper-lower).clamp_min(0).prod(-1)
    union = a[:,3:].prod(-1)[:,None]+b[:,3:].prod(-1)[None]-intersection
    return intersection/union.clamp_min(1e-12)


def rad_loss(outputs,boxes_rad,ignore_iou=.3):
    raw,anchors = outputs["raw"],outputs["anchors"]
    target = torch.zeros_like(raw[...,:6])
    positive = torch.zeros_like(raw[...,6])
    nr,na,nd = raw.shape[1:4]
    valid_targets = []
    collisions = 0
    for batch,values in enumerate(boxes_rad):
        values = torch.as_tensor(values,device=raw.device,dtype=raw.dtype).reshape(-1,6)
        if len(values) and (not torch.isfinite(values).all() or (values[:,:3] < 0).any() or (values[:,:3] > 1).any() or (values[:,3:] <= 0).any()):
            raise ValueError("boxes_rad require normalized finite centers and positive extents")
        valid_targets.append(values)
        order = torch.argsort(values[:,3:].prod(-1),descending=True,stable=True)
        for box in values[order]:
            cells = torch.minimum((box[:3]*raw.new_tensor([nr,na,nd])).long(),torch.tensor([nr-1,na-1,nd-1],device=raw.device))
            intersection = torch.minimum(box[None,3:],anchors).prod(-1)
            match = intersection/(box[3:].prod()+anchors.prod(-1)-intersection).clamp_min(1e-12)
            anchor = int(match.argmax())
            r,a,d = map(int,cells)
            collisions += int(positive[batch,r,a,d,anchor] != 0)
            target[batch,r,a,d,anchor] = box
            positive[batch,r,a,d,anchor] = 1
    prediction,_ = decode_rad(outputs)
    scale = raw.new_tensor(outputs["rad_shape"])
    regression = (((prediction[...,:3]-target[...,:3])*scale)**2 +
                  ((prediction[...,3:]*scale).clamp_min(1e-12).sqrt()-(target[...,3:]*scale).clamp_min(0).sqrt())**2).sum(-1)
    box_loss = (regression*positive).sum()/len(raw)
    ignore = torch.ones_like(positive)
    for b,truth in enumerate(valid_targets):
        if len(truth):
            overlap = iou_rad(prediction[b].detach(),truth).max(-1).values.reshape_as(positive[b])
            ignore[b] = (overlap < ignore_iou).to(raw.dtype)
    probability = raw[...,6].sigmoid()
    objectness = (((positive-probability)**2)*F.binary_cross_entropy_with_logits(raw[...,6],positive,reduction="none")*
                  (positive+.01*(1-positive)*ignore)).sum()/len(raw)
    classification = (F.binary_cross_entropy_with_logits(raw[...,7],torch.ones_like(positive),reduction="none")*positive).sum()/len(raw)
    return {"total":.1*box_loss+objectness+classification,"box":.1*box_loss,"objectness":objectness,
            "classification":classification,"collisions":collisions}


def _collate(items):
    if any("radar" not in v or "boxes_rad" not in v for v in items):
        raise ValueError("RAD pretraining requires native radar and canonical normalized boxes_rad for every sample")
    return {"radar":torch.stack([torch.as_tensor(v["radar"]).float() for v in items]),
            "boxes_rad":[torch.as_tensor(v["boxes_rad"]).float().reshape(-1,6) for v in items]}


def _validation_loss(model,loader,device):
    model.eval()
    validation,n = 0.,0
    with torch.no_grad():
        for batch in loader:
            loss = rad_loss(model(batch["radar"].to(device)),batch["boxes_rad"])["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite RAD validation loss")
            validation += float(loss)*len(batch["radar"])
            n += len(batch["radar"])
    return validation/n


def _matching_prior_best(resume,payload):
    for path in (Path(resume).parent/"best.pt",Path(resume)):
        if not path.is_file():
            continue
        other = torch.load(path,map_location="cpu",weights_only=True)
        if other.get("format_version") != "raddet_rad_v2" or other.get("model_config") != payload["model_config"]:
            continue
        history = other.get("history",[])
        if history and history == payload.get("history",[])[:len(history)] and history[-1]["val_loss"] == payload.get("best_val_loss"):
            return path
    return None


def train_rad_backbone(train_dataset,val_dataset,output_dir,epochs=1,batch_size=3,max_steps=None,
                       learning_rate=1e-4,device="cpu",seed=2024,num_threads=1,resume=None,
                       allow_sensor_change=False):
    """Run additional epochs or continue from a restricted state checkpoint.

    max_steps bounds the TOTAL updates including the resumed history. Resume
    restores model, optimizer, scheduler and best weights when available, starts
    at the next epoch and reseeds shuffle; it is not bitwise minibatch recovery.
    """
    from .training import _plain,_dataset_radar_config,compare_sensor_sampling,enforce_sensor_sampling,_save_checkpoint
    if not len(train_dataset) or not len(val_dataset) or train_dataset is val_dataset:
        raise ValueError("RAD pretraining needs distinct nonempty training/validation datasets")
    if getattr(train_dataset,"split","train") != "train" or getattr(val_dataset,"split","val") == "test":
        raise ValueError("RAD pretraining/model selection must not use the held-out test split")
    if dict(getattr(train_dataset,"normalization",{})) != dict(getattr(val_dataset,"normalization",{})):
        raise ValueError("RAD pretrain/validation normalization differs")
    train_ids = {str(row["id"]) for row in getattr(train_dataset,"samples",[]) if isinstance(row,dict) and "id" in row}
    val_ids = {str(row["id"]) for row in getattr(val_dataset,"samples",[]) if isinstance(row,dict) and "id" in row}
    if train_ids & val_ids:
        raise ValueError("RAD training and validation frame IDs overlap")
    if epochs < 1 or batch_size < 1 or num_threads < 1 or learning_rate <= 0 or (max_steps is not None and max_steps < 1):
        raise ValueError("Invalid RAD pretraining settings")
    torch.set_num_threads(num_threads)
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    shape = getattr(train_dataset,"radar_shape",None) or train_dataset[0]["radar"].shape
    checkpoint = torch.load(resume,map_location=device,weights_only=True) if resume is not None else None
    normalization = _plain(dict(getattr(train_dataset,"normalization",{})))
    if checkpoint is not None:
        if checkpoint.get("format_version") != "raddet_rad_v2" or tuple(checkpoint["model_config"]["input_shape"]) != tuple(shape):
            raise ValueError("RAD resume checkpoint type or native input shape differs")
        if checkpoint["normalization"] != normalization:
            raise ValueError("RAD resume normalization differs")
        model = RADDetRAD(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        model = RADDetRAD(shape).to(device)
    dataset_meta = _plain(getattr(train_dataset,"get_meta",lambda:{})())
    radar_config = _dataset_radar_config(train_dataset)
    if radar_config is not None:
        dataset_meta["radar_config"] = radar_config
    validation_sensor = compare_sensor_sampling(dataset_meta,val_dataset)
    enforce_sensor_sampling(model,validation_sensor,val_dataset,allow_sensor_change)
    resume_sensor = None
    if checkpoint is not None:
        resume_sensor = compare_sensor_sampling(checkpoint.get("dataset_meta",{}),train_dataset)
        enforce_sensor_sampling(model,resume_sensor,train_dataset,allow_sensor_change)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:.96**max((step-60000)//10000,0))
    if checkpoint is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        for group,last_lr in zip(optimizer.param_groups,scheduler.get_last_lr()):
            group["lr"] = last_lr
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed),collate_fn=_collate,num_workers=0)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,collate_fn=_collate,num_workers=0)
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    if checkpoint is None and ((out/"last.pt").exists() or (out/"best.pt").exists()):
        raise FileExistsError("RAD output already contains checkpoints; use resume or a new output directory")
    best,history,step,start_epoch = float("inf"),[],0,0
    selection_scope = "complete_run"
    if checkpoint is not None:
        history = list(checkpoint["history"])
        step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint.get("epoch",history[-1]["epoch"] if history else -1))+1
        prior = _matching_prior_best(resume,checkpoint)
        if prior is not None:
            best = checkpoint["best_val_loss"]
            if prior.resolve() != (out/"best.pt").resolve():
                shutil.copy2(prior,out/"best.pt")
            selection_scope = "continued_run_with_historical_best_weights"
        else:
            selection_scope = "resumed_segment_only_historical_best_weights_unavailable"
    if max_steps is not None and step >= max_steps:
        raise ValueError("max_steps must exceed resumed RAD update count")
    for epoch in range(start_epoch,start_epoch+epochs):
        model.train()
        total,count = 0.,0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = rad_loss(model(batch["radar"].to(device)),batch["boxes_rad"])["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite RAD pretraining loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach())*len(batch["radar"])
            count += len(batch["radar"])
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        validation = _validation_loss(model,val_loader,device)
        history.append({"epoch":epoch,"global_step":step,"train_loss":total/count,"val_loss":validation})
        improved = validation < best
        best = min(best,validation)
        payload = {"format_version":"raddet_rad_v2","model_config":model.export_config,"model_state":model.state_dict(),
                   "optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),
                   "epoch":epoch,"global_step":step,"history":history,"best_val_loss":best,
                   "normalization":normalization,"dataset_meta":dataset_meta,
                   "best_selection_scope":selection_scope,
                   "validation_sensor_comparison":validation_sensor,"resume_sensor_comparison":resume_sensor,
                   "allow_sensor_change":bool(allow_sensor_change),
                   "training_recipe":{"optimizer":"Adam","effective_parameter_groups":[{"learning_rate":float(g["lr"]),"weight_decay":float(g["weight_decay"])} for g in optimizer.param_groups],
                                      "scheduler":"0.96_per10000_after60000_updates","gradient_clip_norm":10,"resume_is_bitwise":False},
                   "task_domain":"normalized_range_azimuth_doppler_not_cartesian",
                   "published_result_reproduced":False}
        _save_checkpoint(payload,out/"last.pt")
        if improved:
            _save_checkpoint(payload,out/"best.pt")
        (out/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        if max_steps is not None and step >= max_steps:
            break
    return out/"best.pt",history


def load_pretrained_raddet_backbone(model,path,freeze=True,normalization=None,dataset_meta=None):
    if not isinstance(model,RADDetBEV):
        raise ValueError("Only RADDet Cartesian models accept a RAD-pretrained backbone")
    payload = torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("format_version") != "raddet_rad_v2" or tuple(payload["model_config"]["input_shape"]) != model.input_shape:
        raise ValueError("RAD backbone checkpoint input geometry/type differs")
    if normalization is not None and payload.get("normalization",{}) != normalization:
        raise ValueError("Pretraining normalization differs from Cartesian training data")
    if dataset_meta is not None:
        from .training import _plain
        source = _plain(payload.get("dataset_meta",{}))
        dataset_meta = _plain(dataset_meta)
        keys = ("shape_native","representation","range_step","range_min","velocity_step","velocity_min_native",
                "range_reversed","doppler_reversed","doppler_positive","azimuth_reversed","azimuth_coordinate","angular_min","angular_step")
        previous,following = source.get("native_profile"),dataset_meta.get("native_profile")
        if isinstance(previous,dict) or isinstance(following,dict):
            if not isinstance(previous,dict) or not isinstance(following,dict) or any(k not in previous or k not in following or previous[k] != following[k] for k in keys):
                raise ValueError("Pretraining native tensor axes/representation differ from Cartesian dataset")
        else:
            # Compare coordinates, not tuple/list serialization or noise/path
            # nuisance settings. The training/evaluation helper uses this same
            # declared sampling signature.
            from .training import _SENSOR_KEYS
            before,after = source.get("radar_config"),dataset_meta.get("radar_config")
            if not isinstance(before,dict) or not isinstance(after,dict) or any(k not in before or k not in after or before[k] != after[k] for k in _SENSOR_KEYS):
                raise ValueError("Pretraining radar coordinates differ from Cartesian dataset or lack provenance")
    state = {k[len("backbone."):]:v for k,v in payload["model_state"].items() if k.startswith("backbone.")}
    model.backbone.load_state_dict(state,strict=True)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(not freeze)
    if freeze:
        model.backbone.eval()
    return {"source":str(path),"global_step":payload["global_step"],"frozen":bool(freeze)}
