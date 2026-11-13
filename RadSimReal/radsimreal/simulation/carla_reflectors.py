"""CARLA semantic-LiDAR captures -> radar reflection points (RadSimReal Sec. 3 interface).

Each captured ray hit is one surface patch. Its area is r^2 * dOmega / cos(theta), the
patch a ray of solid angle dOmega intercepts on a surface at incidence theta. Patch
backscatter uses the package's SurfaceMaterial model (Fresnel power, roughness,
diffuse/specular lobes) with per-semantic-class materials from a written specification:
ITU-R P.2040-1 Table 3 parameters evaluated at 77 GHz where the table covers the
material (concrete, brick, wood, glass, metal), and disclosed assumptions otherwise.
Nothing here is fitted to radar data; only the single global relative gain is
fitted separately on TRAIN frames (see scripts/generate_carla_dataset.py).

Received power  P = gain * sigma * (reference_range / r)^4,  sigma = rho(theta) * area.
Complex field   sqrt(P) * exp(j(-4 pi r / lambda + 2 pi U)), U ~ Uniform (rough-surface phase).
Single bounce; no multipath. Points keep their actor's velocity (wheels/limbs not modelled).
"""
from __future__ import annotations

import numpy as np

from .materials import LIGHT_SPEED_M_S, SurfaceMaterial
from .reflections import ReflectionPoints

F_GHZ = 77.0


def _itu(name, a, c, d, roughness_m, diffuse_fraction, two_sided=False):
    """ITU-R P.2040-1: eps' = a f^b (b=0 here), sigma = c f^d [S/m], f in GHz."""
    return SurfaceMaterial(name=name, relative_permittivity_real=a, conductivity_s_per_m=c * F_GHZ ** d,
                           roughness_m=roughness_m, diffuse_fraction=diffuse_fraction, two_sided=two_sided)


METAL = SurfaceMaterial(name="metal", relative_permittivity_real=1.0, conductivity_s_per_m=1e7,
                        roughness_m=0.0002, diffuse_fraction=0.35)
CONCRETE = _itu("concrete_ITU", 5.24, 0.0462, 0.7822, 0.001, 0.6)
ASPHALT = _itu("asphalt_as_concrete_ITU", 5.24, 0.0462, 0.7822, 0.002, 0.8)
BRICK = _itu("brick_ITU", 3.91, 0.0238, 0.16, 0.001, 0.6)
WOOD = _itu("wood_ITU", 1.99, 0.0047, 1.0718, 0.001, 0.6)
GLASS = _itu("glass_ITU", 6.31, 0.0036, 1.3394, 0.0001, 0.1)
# Not covered by ITU-R P.2040 at 77 GHz -> disclosed assumptions:
VEGETATION = SurfaceMaterial(name="vegetation_assumed", relative_permittivity_real=5.0, conductivity_s_per_m=1.0,
                             roughness_m=0.003, diffuse_fraction=0.95, two_sided=True)
GROUND = SurfaceMaterial(name="ground_assumed", relative_permittivity_real=10.0, conductivity_s_per_m=5.0,
                         roughness_m=0.003, diffuse_fraction=0.9)
HUMAN = SurfaceMaterial(name="skin_assumed_77GHz", relative_permittivity_real=6.4, conductivity_s_per_m=38.5,
                        roughness_m=0.0005, diffuse_fraction=0.6)
WATER = SurfaceMaterial(name="water_assumed", relative_permittivity_real=10.0, conductivity_s_per_m=60.0,
                        roughness_m=0.0005, diffuse_fraction=0.2)



class MeasuredBackscatter:
    """Backscatter coefficient sigma0(theta) from a published measurement table (dB vs incidence angle).

    Linear interpolation in dB; below the first measured angle the first value is held, beyond the
    last the final dB slope is continued. Polarization TE -> HH, TM -> VV.
    """

    def __init__(self, name, source, angles_deg, sigma0_db_hh, sigma0_db_vv, note=""):
        self.name, self.source, self.note = name, source, note
        self.angles = np.asarray(angles_deg, float)
        self.table = {"TE": np.asarray(sigma0_db_hh, float), "TM": np.asarray(sigma0_db_vv, float)}

    def backscatter_coefficient(self, cos_incidence, frequency_hz=77e9, polarization="TE"):
        theta = np.degrees(np.arccos(np.clip(np.asarray(cos_incidence, float), 0, 1)))
        db = self.table[polarization.upper()]
        out = np.interp(theta, self.angles, db)
        slope = (db[-1] - db[-2]) / (self.angles[-1] - self.angles[-2])
        beyond = theta > self.angles[-1]
        out[beyond] = db[-1] + slope * (theta[beyond] - self.angles[-1])
        return 10 ** (out / 10)

    def to_dict(self):
        return {"name": self.name, "source": self.source, "angles_deg": self.angles.tolist(),
                "sigma0_db_hh": self.table["TE"].tolist(), "sigma0_db_vv": self.table["TM"].tolist(), "note": self.note}


# Dry asphalt, read from Sarabandi, Li & Nashashibi, IEEE T-AP 45(11) 1997, Fig. 5(a)/(b): the model
# curve that tracks the measured points of both sites (94 GHz, the nearest published measurement found;
# reading uncertainty about +-1 dB). Used for paved ground: road, road markings, sidewalk.
ASPHALT_MEASURED = MeasuredBackscatter(
    "dry_asphalt_measured_94GHz_Sarabandi1997",
    "Sarabandi, Li, Nashashibi, IEEE Trans. Antennas Propag. 45(11):1679-1688, 1997, Fig. 5",
    [70, 72, 74, 76, 78, 80, 82, 84, 86, 88],
    [-17.5, -18.3, -19.2, -20.7, -22.4, -24.5, -27.0, -30.5, -35.5, -43.5],
    [-13.0, -13.5, -14.3, -15.2, -16.3, -17.8, -19.6, -22.6, -27.0, -34.5],
    note="measured 70-88 deg at 94 GHz; held at the 70 deg value below 70 deg; last slope continued above 88 deg")

# CARLA 0.9.15 semantic tags (CityScapes-based).
TAG_MATERIAL_V1 = {
    0: None, 1: ASPHALT, 2: CONCRETE, 3: CONCRETE, 4: BRICK, 5: METAL, 6: METAL, 7: METAL, 8: METAL,
    9: VEGETATION, 10: GROUND, 11: None, 12: HUMAN, 13: HUMAN, 14: METAL, 15: METAL, 16: METAL, 17: METAL,
    18: METAL, 19: METAL, 20: CONCRETE, 21: METAL, 22: CONCRETE, 23: WATER, 24: ASPHALT, 25: GROUND,
    26: CONCRETE, 27: METAL, 28: METAL,
}
# v2: paved ground surfaces use the measured asphalt curve instead of the assumed rough-surface model.
TAG_MATERIAL = {**TAG_MATERIAL_V1, 1: ASPHALT_MEASURED, 2: ASPHALT_MEASURED, 24: ASPHALT_MEASURED}
MATERIAL_SETS = {"v1_assumed": TAG_MATERIAL_V1, "v2_measured_road": TAG_MATERIAL}
TAG_NAMES = {0: "unlabeled", 1: "road", 2: "sidewalk", 3: "building", 4: "wall", 5: "fence", 6: "pole",
             7: "traffic_light", 8: "traffic_sign", 9: "vegetation", 10: "terrain", 11: "sky", 12: "pedestrian",
             13: "rider", 14: "car", 15: "truck", 16: "bus", 17: "train", 18: "motorcycle", 19: "bicycle",
             20: "static", 21: "dynamic", 22: "other", 23: "water", 24: "road_line", 25: "ground", 26: "bridge",
             27: "rail_track", 28: "guard_rail"}


def material_table(materials=None):
    materials = TAG_MATERIAL if materials is None else materials
    return {TAG_NAMES[t]: (m.to_dict() if m is not None else None) for t, m in materials.items()}


def capture_to_reflections(capture, ray_solid_angle_sr, radar_config, rng, min_cos=0.05, min_range=0.3, materials=None):
    """capture: mapping with xyz (N,3) sensor frame, cos_inc, tag, object_idx, velocity."""
    xyz = np.asarray(capture["xyz"], dtype=float)
    velocity = np.asarray(capture["velocity"], dtype=float)
    cosine = np.clip(np.asarray(capture["cos_inc"], dtype=float), 0, 1)
    tags = np.asarray(capture["tag"], dtype=int)
    object_idx = np.asarray(capture["object_idx"], dtype=np.int64)
    distance = np.linalg.norm(xyz, axis=1)
    keep = (distance >= min_range) & (cosine > 0)
    materials = TAG_MATERIAL if materials is None else materials
    keep &= np.array([materials.get(int(t)) is not None for t in tags]) if len(tags) else keep
    xyz, velocity, cosine, tags, object_idx, distance = (a[keep] for a in (xyz, velocity, cosine, tags, object_idx, distance))
    rho = np.zeros(len(tags))
    for t in np.unique(tags):
        sel = tags == t
        rho[sel] = materials[int(t)].backscatter_coefficient(cosine[sel], F_GHZ * 1e9, radar_config.polarization)
    area = distance ** 2 * ray_solid_angle_sr / np.maximum(cosine, min_cos)
    sigma = rho * area
    power = radar_config.scattering_gain * sigma * (radar_config.reference_range / distance) ** 4
    if radar_config.representation in ("field", "complex"):
        wavelength = LIGHT_SPEED_M_S / (F_GHZ * 1e9)
        phase = -4 * np.pi * distance / wavelength + 2 * np.pi * rng.random(len(power))
        coefficients = np.sqrt(power) * np.exp(1j * phase)
    else:
        coefficients = power
    nonzero = power > 0
    return ReflectionPoints(xyz[nonzero], velocity[nonzero], coefficients[nonzero], object_idx[nonzero], metadata={
        "source": "CARLA semantic LiDAR ray hits (single bounce)", "ray_solid_angle_sr": float(ray_solid_angle_sr),
        "material_spec": "ITU-R P.2040-1 at 77 GHz where covered; disclosed assumptions otherwise",
        "patch_area": "r^2 dOmega / max(cos, %.2f)" % min_cos, "phase": "two-way path + uniform rough-surface phase",
        "tags": np.bincount(tags[nonzero], minlength=29).tolist()})


LABEL_CLASSES = ("car", "truck", "bus")  # RADDet's car+truck+bus, matching the paper's car counts


def vehicle_objects(meta, reflections, radar_config, grid, min_points=5):
    """Labels: car/truck/bus with >= min_points in-domain returns and footprint inside the BEV grid.

    Footprints partly outside the grid are not clipped; they are listed as excluded.
    """
    from ..radar.coordinates import points_to_indices
    _, valid = points_to_indices(reflections.xyz, reflections.velocity, radar_config)
    objects, excluded = [], []
    for v in meta["vehicles"]:
        if v["label_class"] not in LABEL_CLASSES:
            continue
        count = int(np.sum(valid & (reflections.object_ids == v["actor_id"])))
        if count < min_points:
            continue
        box = np.asarray(v["footprint_xyxy"], dtype=float)
        normalized = grid.normalize_boxes(box)
        if np.any(normalized < 0) or np.any(normalized > 1):
            excluded.append({"actor_id": v["actor_id"], "reason": "footprint leaves BEV grid", "box_bev": box.tolist()})
            continue
        objects.append({"object_id": int(v["actor_id"]), "label": "car", "carla_class": v["label_class"],
                        "type_id": v["type_id"], "box_bev": box.tolist(), "in_domain_reflection_count": count,
                        "label_scope": "CARLA vehicle bounding-box ground footprint, axis-aligned in sensor frame",
                        "velocity_sensor": v["velocity_sensor"], "has_simulated_return": True})
    return objects, excluded
