"""Resumable on-cluster pilot: native I/O -> calibration -> simulation -> models.

No network/SSH connection is made. This executes wherever the user's native
files reside. Unconfirmed automatic selections remain explicitly provisional.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from .workflows import (make_reader, prepare_native, inspect_native, propose_calibration,
                        reader_from_prepared)
from .calibration import calibrate_native, json_write
from ..config import load_config
from ..datasets.generation_v2 import generate_bev_dataset, fit_relative_scattering_gain
from ..datasets.bev_dataset import BEVDataset
from ..bev.geometry import BEVGrid
from ..bev.training import train_bev, load_bev_checkpoint
from ..bev.evaluation import evaluate_bev
from .validation import validate_distribution


def run_native_pilot(args):
    started = time.perf_counter()
    reader = make_reader(args.root, args.dataset, profile=args.profile, representation=args.representation,
                         trusted_labels=args.trusted_labels, annotation_path=args.annotations,
                         reference_cartesian=args.reference_cartesian)
    out = Path(args.out).expanduser().resolve()
    try:
        out.relative_to(reader.root)
    except ValueError:
        pass
    else:
        raise ValueError("Choose a run output outside the original dataset directory")
    options = {k: v for k, v in vars(args).items() if k not in ("run", "resume_run") and not callable(v)}
    options["profile_resolved"] = reader.profile.to_dict()
    options["workflow_schema_version"] = 2
    if args.selection:
        options["selection_sha256"] = hashlib.sha256(Path(args.selection).read_bytes()).hexdigest()
    # JSON storage converts tuples to lists; compare the same canonical form.
    options = json.loads(json.dumps(options))
    if out.exists() and any(out.iterdir()):
        if not args.resume_run or not (out / "run_inputs.json").is_file():
            raise FileExistsError("Run output must be empty, or use --resume-run with its original recorded settings")
        previous = json.loads((out / "run_inputs.json").read_text(encoding="utf-8"))
        if previous != options:
            ignored = {"selection", "selection_sha256", "allow_provisional"}
            comparable_previous = {k: v for k, v in previous.items() if k not in ignored}
            comparable_current = {k: v for k, v in options.items() if k not in ignored}
            transition_allowed = (not (out / "calibration/sensor_config.json").exists()
                                  and previous.get("selection") is None
                                  and comparable_previous == comparable_current)
            if not transition_allowed:
                raise ValueError("Resume-run settings differ; use a new output directory for a changed experiment")
            transitions = json.loads((out / "run_input_transitions.json").read_text()) if (out / "run_input_transitions.json").exists() else []
            transitions.append({"reason": "first calibration selection supplied after candidate inspection",
                                "previous": {k: previous.get(k) for k in ignored},
                                "current": {k: options.get(k) for k in ignored}})
            json_write(out / "run_input_transitions.json", transitions)
    out.mkdir(parents=True, exist_ok=True)
    json_write(out / "run_inputs.json", options)
    report = {"dataset": args.dataset, "native_root": str(reader.root), "stages": {},
              "published_result_reproduced": False, "execution_host_has_native_data": True}

    def done(stage, value):
        report["stages"][stage] = value
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        json_write(out / "run_report.json", report)
        print(f"Completed {stage}", flush=True)

    if not (out / "inspection/inspection.json").exists():
        inspect_native(reader, out / "inspection", max_frames=3)
    done("inspection", {"record_count": len(reader.records), "profile": "inspection/profile_reference.json"})
    prepared = out / "prepared_native"
    if not (prepared / "manifest.json").exists():
        grid = BEVGrid.from_dict(json.loads(Path(args.grid).read_text())) if args.grid else None
        automatic = args.split_file is None and args.val_sequences is None and args.test_sequences is None
        prepare_native(reader, prepared, val_sequences=args.val_sequences, test_sequences=args.test_sequences,
                       split_file=args.split_file, auto_split=automatic, grid=grid,
                       fit_normalization_frames=16, verify_all_labels=True)
    reader, manifest = reader_from_prepared(prepared)
    if args.raddet_pretrain and args.train_real_baseline and args.dataset == "carrada" and "raddet" in args.models:
        raise ValueError("Public CARRADA labels lack native 3D RAD boxes. A matched two-stage real-only RADDet baseline is unavailable; omit --raddet-pretrain for both comparison arms or omit the real-only baseline.")
    done("native_preparation", {"counts": manifest["counts"], "split_source": manifest["split_source"],
                                  "label_domain": manifest["label_domain"]})
    selection = args.selection
    if selection is None:
        template = out / "candidates/selection_template.json"
        if not template.exists():
            propose_calibration(prepared, out / "candidates", max_frames=16, half_widths=args.half_widths)
        if not args.allow_provisional:
            report["status"] = "needs_verified_calibration_selection"
            report["next_action"] = "Inspect candidate PNGs and supply a verified selection JSON, or explicitly choose --allow-provisional for an unvalidated pipeline pilot."
            done("candidate_selection", {"selection_template": str(template)})
            print(report["next_action"], flush=True)
            return report
        selection = template
    selected = json.loads(Path(selection).read_text(encoding="utf-8"))
    confirmed = all(selected.get(k) is True for k in ("target_is_isolated", "sensor_target_stationary", "domain_verified"))
    if not confirmed and not args.allow_provisional:
        raise ValueError("Selection target assumptions are unconfirmed; verify them or explicitly allow a provisional pilot")
    sensor = out / "calibration/sensor_config.json"
    if sensor.exists():
        used_selection = json.loads((out / "calibration/selection.json").read_text(encoding="utf-8"))
        if used_selection != selected:
            raise ValueError("Calibration selection changed since sensor products were generated; use a new run directory to recalibrate dependent stages")
    else:
        calibrate_native(reader, selected, out / "calibration",
            allowed_ids=set(manifest.get("calibration_record_ids", [r["id"] for r in manifest["splits"]["train"]])),
            noise_model="empirical", phase_align=reader.profile.representation == "complex")
    report["calibration_selection_status"] = "confirmed_input_assumptions" if confirmed else "provisional_unconfirmed"
    done("calibration", {"sensor_config": str(sensor), "selection_status": report["calibration_selection_status"]})
    if args.fit_gain:
        gain_config = out / "calibration/gain_fitted_config.json"
        if not gain_config.exists():
            fit_relative_scattering_gain(prepared, sensor, gain_config, real_frames=16,
                                         synthetic_frames=4, max_samples=args.max_samples,
                                         vehicle_assets_path=args.vehicle_assets)
        sensor = gain_config
        done("relative_gain_fit", {"source": "unlabeled_training_quantiles_only", "absolute_RCS": False})
    generated = out / "synthetic"
    if not (generated / "manifest.json").exists():
        generate_bev_dataset(generated, load_config(sensor), {"train": args.train, "val": args.val, "test": args.test},
            grid=BEVGrid.from_dict(manifest["bev_grid"]), normalization=manifest["normalization"],
            native_profile=reader.profile, max_samples=args.max_samples, vehicle_assets_path=args.vehicle_assets)
    done("simulation", {"counts": {"train": args.train, "val": args.val, "test": args.test}, "native_export": str(generated / "native_export")})
    if manifest["splits"]["test"] and args.test:
        diagnostic = validate_distribution(prepared, generated, out / "distribution_validation", max_frames=16)
        done("distribution_diagnostics", {k: v for k, v in diagnostic.items() if k != "datasets"})
    budget = None if args.max_steps == 0 else args.max_steps

    def progress(directory):
        directory = Path(directory)
        last = directory / "last.pt"
        if not last.exists():
            return None, args.epochs, False
        payload = torch.load(last, map_location="cpu", weights_only=True)
        best = directory / "best.pt"
        expected_best = payload.get("best_val_loss")
        history = payload.get("history", [])
        best_matches = False
        if best.exists():
            saved_best = torch.load(best, map_location="cpu", weights_only=True)
            saved_history = saved_best.get("history", [])
            best_matches = bool(saved_history and saved_history[-1].get("val_loss") == expected_best
                                and saved_history == history[:len(saved_history)])
        if not best_matches:
            if history and history[-1].get("val_loss") == expected_best:
                replacement = directory / "best.pt.recovering"
                shutil.copy2(last, replacement)
                replacement.replace(best)
            else:
                raise ValueError("Historical best weights are missing or inconsistent with last.pt; restore the recorded best checkpoint before resuming this run")
        completed_epochs = len(payload.get("history", []))
        complete = (budget is not None and int(payload.get("global_step", 0)) >= budget) or completed_epochs >= args.epochs
        return last, max(1, args.epochs - completed_epochs), complete

    for model_name in args.models:
        model_dir = out / "models" / f"synthetic_{model_name}"
        train_data = BEVDataset(generated, "train", include_radar=model_name == "raddet")
        val_data = BEVDataset(generated, "val", normalization=train_data.normalization, include_radar=model_name == "raddet")
        pretrained = None
        if model_name == "raddet" and args.raddet_pretrain:
            from ..bev.rad_pretraining import train_rad_backbone
            pretrain_dir = out / "models/raddet_3d_pretraining"
            pretrained = pretrain_dir / "best.pt"
            resume, remaining_epochs, complete = progress(pretrain_dir)
            if not complete:
                train_rad_backbone(train_data, val_data, pretrain_dir, epochs=remaining_epochs,
                    batch_size=args.batch_size, max_steps=budget, device=args.device, resume=resume)
        checkpoint = model_dir / "best.pt"
        resume, remaining_epochs, complete = progress(model_dir)
        if not complete:
            checkpoint, _ = train_bev(train_data, val_data, model_dir, model_name=model_name,
                epochs=remaining_epochs, batch_size=args.batch_size, max_steps=budget, base_channels=args.base_channels,
                device=args.device, pretrained_backbone=pretrained, freeze_backbone=pretrained is not None, resume=resume)
        model, payload = load_bev_checkpoint(checkpoint, args.device)
        if manifest["splits"]["test"]:
            actual_test = BEVDataset(prepared, "test", normalization=payload["normalization"],
                                     include_radar=model.requires_radar, grid=model.grid)
            metrics = evaluate_bev(checkpoint, actual_test, batch_size=args.batch_size, device=args.device,
                output=model_dir / "native_test_metrics.json", allow_label_domain_change=True,
                matching_policy=args.matching_policy, ap_aggregation=args.ap_aggregation)
            done(f"synthetic_{model_name}_native_test", {k: v for k, v in metrics.items() if k not in ("curves", "image_ids")})
        if args.train_real_baseline:
            baseline = out / "models" / f"native_{model_name}"
            actual_train = BEVDataset(prepared, "train", include_radar=model_name == "raddet")
            actual_val = BEVDataset(prepared, "val", normalization=actual_train.normalization, include_radar=model_name == "raddet")
            real_pretrained = None
            if model_name == "raddet" and args.raddet_pretrain:
                from ..bev.rad_pretraining import train_rad_backbone
                real_pretrain_dir = out / "models/native_raddet_3d_pretraining"
                real_pretrained = real_pretrain_dir / "best.pt"
                resume_rad, remaining_rad_epochs, rad_complete = progress(real_pretrain_dir)
                if not rad_complete:
                    train_rad_backbone(actual_train, actual_val, real_pretrain_dir, epochs=remaining_rad_epochs,
                        batch_size=args.batch_size, max_steps=budget, device=args.device, resume=resume_rad)
            real_checkpoint = baseline / "best.pt"
            resume, remaining_epochs, complete = progress(baseline)
            if not complete:
                real_checkpoint, _ = train_bev(actual_train, actual_val, baseline, model_name=model_name,
                    epochs=remaining_epochs, batch_size=args.batch_size, max_steps=budget,
                    base_channels=args.base_channels, device=args.device, resume=resume,
                    pretrained_backbone=real_pretrained, freeze_backbone=real_pretrained is not None)
            if manifest["splits"]["test"]:
                real_metrics = evaluate_bev(real_checkpoint, BEVDataset(prepared, "test", include_radar=model_name == "raddet"),
                    batch_size=args.batch_size, device=args.device, output=baseline / "native_test_metrics.json",
                    matching_policy=args.matching_policy, ap_aggregation=args.ap_aggregation)
                done(f"native_{model_name}_baseline", {k: v for k, v in real_metrics.items() if k not in ("curves", "image_ids")})
    report["status"] = "pilot_completed" if budget is not None else "requested_training_completed"
    report["interpretation"] = "Execution on supplied inputs; model convergence, sensor realism and published AP require separate evidence."
    done("finished", {"max_steps_per_model": budget, "epochs_requested": args.epochs})
    print(f"Saved complete run report to {out / 'run_report.json'}", flush=True)
    return report
