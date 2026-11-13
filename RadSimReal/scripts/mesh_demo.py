"""Render the v2 mesh/PSF simulator without datasets or learned weights."""
import argparse
import json
from pathlib import Path
import numpy as np
from radsimreal.config import load_config, save_config
from radsimreal.datasets.generation_v2 import _scene_reflections
from radsimreal.datasets.synthetic import calibrated_psf, scene_annotations
from radsimreal.radar import simulate
from radsimreal.simulation import sample_scene, save_reflections
from radsimreal.simulation.assets import load_vehicle_assets
from radsimreal.visualization import plot_demo, plot_radar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cartesian_demo.yaml")
    parser.add_argument("--vehicle-assets", default="configs/assets/vehicle_assets.json")
    parser.add_argument("--out", default="runs/mesh_demo")
    parser.add_argument("--max-samples", type=int, default=2000)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty demo output directory")
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    rng = np.random.default_rng(config.seed)
    scene = sample_scene(config.radar, config.scene, rng)
    library = load_vehicle_assets(args.vehicle_assets) if args.vehicle_assets else None
    reflections = _scene_reflections(scene, config, rng, max_samples=args.max_samples,
                                     vehicle_assets=library)
    result = simulate(reflections, config.radar, calibrated_psf(config.radar), rng)
    annotations = scene_annotations(scene, config.radar, reflections)
    boxes = [obj["box"] for obj in annotations]
    np.savez_compressed(out / "radar_arrays.npz", tensor=result.tensor,
                        image=result.image, psf=result.psf)
    save_reflections(out / "reflections.npz", reflections)
    save_config(config, out / "config.json")
    (out / "scene.json").write_text(json.dumps({"source": "synthetic_mesh_demo",
        "scene": scene.to_dict(), "radar_view_annotations": annotations,
        "simulation": result.metadata, "reflection_metadata": reflections.metadata,
        "published_result_reproduced": False}, indent=2, allow_nan=False), encoding="utf-8")
    plot_demo(scene, reflections, result, config.radar, out / "mesh_overview.png", boxes)
    plot_radar(result.image, config.radar, out / "radar.png", boxes,
               title="Synthetic CAD mesh radar — uncalibrated illustrative sensor")
    print(f"Saved mesh geometry/reflections, full tensor and radar figures to {out}")


if __name__ == "__main__":
    main()
