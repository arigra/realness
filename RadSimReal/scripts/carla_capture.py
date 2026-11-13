"""Capture radar reflection points and vehicle labels from a running CARLA server.

Two sensor mountings:
  --mount vehicle  (paper, Sec. 4: "a radar mounted on a vehicle driving in city streets,
                   experiencing both stationary and moving phases"). A forward-facing sensor on
                   the front bumper of an autopilot ego car; traffic-manager driving includes
                   stops at lights/junctions and moving phases. One episode per position index.
  --mount sidewalk (earlier choice, kept for comparison): static sensor on the sidewalk facing the road.

Each frame stores a ray-traced semantic-LiDAR scan restricted to the radar field of view: per-ray
hit point, incidence cosine, CARLA semantic tag and object index, plus per-point velocity RELATIVE
TO THE SENSOR (hit actor velocity minus sensor velocity; static world = minus sensor velocity), and
every other vehicle's footprint/relative velocity/class in the sensor frame.

Sensor frame written here: x right, y forward, z up (metres), matching radsimreal.
Resumable: a position whose output folder has DONE is skipped.

Usage (server already running):
  python scripts/carla_capture.py --mount vehicle --out runs/carla_capture_vehicle --towns Town01 ... \
      --positions-per-town 110 --frames-per-position 20
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import carla

VEHICLE_CLASS = {"car": "car", "van": "car", "truck": "truck", "bus": "bus",
                 "motorcycle": "motorcycle", "bicycle": "bicycle"}
EGO_BLUEPRINT = "vehicle.tesla.model3"


def sensor_basis(transform):
    """World->sensor rotation. CARLA (UE) is left-handed x fwd, y right, z up."""
    fwd = transform.get_forward_vector(); right = transform.get_right_vector(); up = transform.get_up_vector()
    return np.array([[right.x, right.y, right.z], [fwd.x, fwd.y, fwd.z], [up.x, up.y, up.z]])


def to_sensor(points_world, transform):
    loc = transform.location
    R = sensor_basis(transform)
    return (np.asarray(points_world) - [loc.x, loc.y, loc.z]) @ R.T


def vehicle_records(world, sensor_tf, sensor_velocity=(0.0, 0.0, 0.0), skip_ids=()):
    """Vehicle footprints and velocities relative to the (possibly moving) sensor, in the sensor frame."""
    R = sensor_basis(sensor_tf)
    sv = np.asarray(sensor_velocity, dtype=float)
    out = []
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id in skip_ids:
            continue
        verts = np.array([[v.x, v.y, v.z] for v in actor.bounding_box.get_world_vertices(actor.get_transform())])
        local = to_sensor(verts, sensor_tf)
        vel = actor.get_velocity()
        v_local = R @ (np.array([vel.x, vel.y, vel.z]) - sv)
        base = actor.attributes.get("base_type", "").lower() or "car"
        out.append({"actor_id": actor.id, "type_id": actor.type_id, "base_type": base,
                    "label_class": VEHICLE_CLASS.get(base, "car"),
                    "footprint_xyxy": [float(local[:, 0].min()), float(local[:, 1].min()),
                                       float(local[:, 0].max()), float(local[:, 1].max())],
                    "z_range": [float(local[:, 2].min()), float(local[:, 2].max())],
                    "velocity_sensor": v_local.tolist(),
                    "extent": [actor.bounding_box.extent.x, actor.bounding_box.extent.y, actor.bounding_box.extent.z]})
    return out


def actor_velocities(world, sensor_tf, sensor_velocity=(0.0, 0.0, 0.0)):
    """Actor velocity relative to the sensor, in the sensor frame. Key 0 = static world."""
    R = sensor_basis(sensor_tf)
    sv = np.asarray(sensor_velocity, dtype=float)
    table = {0: R @ (-sv)}
    for actor in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.*")):
        v = actor.get_velocity()
        table[actor.id] = R @ (np.array([v.x, v.y, v.z]) - sv)
    return table


def sensor_world_velocity(ego, sensor_tf):
    """Velocity of the sensor point on the rigid ego body: v_c + omega x (s - c). CARLA angular velocity is deg/s."""
    v = ego.get_velocity(); w = ego.get_angular_velocity()
    c = ego.get_transform().location; s = sensor_tf.location
    omega = np.radians([w.x, w.y, w.z])
    return np.array([v.x, v.y, v.z]) + np.cross(omega, [s.x - c.x, s.y - c.y, s.z - c.z])


def sidewalk_positions(world_map, n, rng):
    """Sensor poses on the sidewalk beside random driving lanes, facing the road."""
    wps = [w for w in world_map.generate_waypoints(8.0) if w.lane_type == carla.LaneType.Driving and not w.is_junction]
    rng.shuffle(wps)
    poses = []
    for wp in wps:
        right = wp.get_right_lane()
        # outermost driving lane only, so the sensor stands off the carriageway
        if right is not None and right.lane_type == carla.LaneType.Driving:
            continue
        tf = wp.transform
        r = tf.get_right_vector()
        offset = wp.lane_width / 2 + rng.uniform(1.5, 3.5)
        loc = carla.Location(tf.location.x + r.x * offset, tf.location.y + r.y * offset,
                             tf.location.z + rng.uniform(0.6, 1.4))
        yaw = tf.rotation.yaw - 90 + rng.uniform(-50, 50)  # face back across the road
        poses.append(carla.Transform(loc, carla.Rotation(pitch=0, yaw=yaw, roll=0)))
        if len(poses) >= n:
            break
    return poses


def spawn_ego(client, world, rng, tm_port):
    """Autopilot ego car at a random spawn point (traffic-manager driving: stops and moving phases)."""
    bp = world.get_blueprint_library().find(EGO_BLUEPRINT)
    bp.set_attribute("role_name", "hero")
    spawn = world.get_map().get_spawn_points()
    order = rng.permutation(len(spawn))
    for i in order:
        ego = world.try_spawn_actor(bp, spawn[int(i)])
        if ego is not None:
            ego.set_autopilot(True, tm_port)
            return ego
    raise RuntimeError("no free spawn point for the ego vehicle")


def spawn_traffic(client, world, n_vehicles, n_walkers, rng, tm_port):
    bp = world.get_blueprint_library()
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(int(rng.integers(1 << 30)))
    tm.global_percentage_speed_difference(float(rng.uniform(-30, 30)))
    spawn = world.get_map().get_spawn_points()
    rng.shuffle(spawn)
    vbps = [b for b in bp.filter("vehicle.*")]
    batch = [carla.command.SpawnActor(rng.choice(vbps), sp).then(
        carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)) for sp in spawn[:n_vehicles]]
    vehicles = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]
    walkers, controllers = [], []
    wbps = bp.filter("walker.pedestrian.*")
    locs = [world.get_random_location_from_navigation() for _ in range(n_walkers)]
    batch = [carla.command.SpawnActor(rng.choice(wbps), carla.Transform(l)) for l in locs if l is not None]
    walkers = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]
    cbp = bp.find("controller.ai.walker")
    batch = [carla.command.SpawnActor(cbp, carla.Transform(), w) for w in walkers]
    controllers = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]
    world.tick()
    for c in world.get_actors(controllers):
        c.start(); c.go_to_location(world.get_random_location_from_navigation()); c.set_max_speed(1 + rng.random())
    return vehicles, walkers, controllers


def teardown(client, world, vehicles, walkers, controllers, tm_port):
    """Release traffic-manager control and AI walkers before destroying, then let the server settle."""
    alive = {a.id for a in world.get_actors()}
    for c in world.get_actors([c for c in controllers if c in alive]):
        c.stop()
    client.apply_batch_sync([carla.command.SetAutopilot(v, False, tm_port) for v in vehicles if v in alive], True)
    client.apply_batch_sync([carla.command.DestroyActor(x) for x in controllers if x in alive], True)
    client.apply_batch_sync([carla.command.DestroyActor(x) for x in walkers + vehicles if x in alive], True)
    for _ in range(5):
        world.tick()


def make_lidar(world, tf, args, attach_to=None):
    bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
    for k, v in {"channels": args.channels, "range": args.range, "upper_fov": args.upper_fov,
                 "lower_fov": args.lower_fov, "horizontal_fov": args.horizontal_fov,
                 "rotation_frequency": 1.0 / args.dt,
                 "points_per_second": int(args.channels * args.horizontal_fov / args.az_res / args.dt)}.items():
        bp.set_attribute(k, str(v))
    if attach_to is None:
        return world.spawn_actor(bp, tf)
    return world.spawn_actor(bp, tf, attach_to=attach_to, attachment_type=carla.AttachmentType.Rigid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mount", choices=["vehicle", "sidewalk"], default="vehicle")
    ap.add_argument("--mount-height", type=float, default=0.7,
                    help="vehicle mount: sensor height above the ego origin (ground), metres (paper: not stated)")
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--towns", nargs="+", default=["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD_Opt"])
    ap.add_argument("--positions-per-town", type=int, default=100)
    ap.add_argument("--frames-per-position", type=int, default=20)
    ap.add_argument("--ticks-between-frames", type=int, default=10)
    ap.add_argument("--warmup-ticks", type=int, default=40)
    ap.add_argument("--vehicles", type=int, nargs=2, default=[60, 140])
    ap.add_argument("--walkers", type=int, nargs=2, default=[30, 90])
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--upper-fov", type=float, default=15.0); ap.add_argument("--lower-fov", type=float, default=-15.0)
    ap.add_argument("--horizontal-fov", type=float, default=180.0)
    ap.add_argument("--az-res", type=float, default=0.2, help="degrees between rays in azimuth")
    ap.add_argument("--range", type=float, default=55.0)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "capture_args.json").write_text(json.dumps(vars(args), indent=1))
    client = carla.Client(args.host, args.port); client.set_timeout(120.0)
    solid_angle = math.radians(args.az_res) * math.radians((args.upper_fov - args.lower_fov) / (args.channels - 1))
    for town in args.towns:
        rng = np.random.default_rng([args.seed, sum(map(ord, town))])
        todo = [p for p in range(args.positions_per_town) if not (out / town / f"pos_{p:04d}" / "DONE").exists()]
        if not todo:
            continue
        world = client.get_world()
        if world.get_map().name.split("/")[-1] != town:
            # Map switches hold two maps in memory; under a tight job memory cap start the
            # server directly in the town instead (CarlaUE4.sh /Game/Carla/Maps/<town>).
            world = client.load_world(town)
        settings = world.get_settings(); settings.synchronous_mode = True; settings.fixed_delta_seconds = args.dt
        world.apply_settings(settings)
        poses = None
        if args.mount == "sidewalk":
            poses = sidewalk_positions(world.get_map(), args.positions_per_town, random.Random(int(rng.integers(1 << 30))))
        for p in todo:
            if poses is not None and p >= len(poses):
                break
            pos_dir = out / town / f"pos_{p:04d}"; pos_dir.mkdir(parents=True, exist_ok=True)
            prng = np.random.default_rng([args.seed, sum(map(ord, town)), p])
            ego = None
            if args.mount == "vehicle":
                client.get_trafficmanager(args.tm_port).set_synchronous_mode(True)
                ego = spawn_ego(client, world, prng, args.tm_port)
            vehicles, walkers, controllers = spawn_traffic(client, world, int(prng.integers(*args.vehicles)),
                                                           int(prng.integers(*args.walkers)), prng, args.tm_port)
            if ego is not None:
                # front bumper, centred, facing forward; just ahead of the ego box so rays do not start inside it
                mount_tf = carla.Transform(carla.Location(x=ego.bounding_box.location.x + ego.bounding_box.extent.x + 0.05,
                                                          y=0.0, z=args.mount_height))
                lidar = make_lidar(world, mount_tf, args, attach_to=ego)
            else:
                lidar = make_lidar(world, poses[p], args)
            latest = {}
            lidar.listen(lambda m: latest.__setitem__("m", m))
            for _ in range(args.warmup_ticks):
                world.tick()
            for f in range(args.frames_per_position):
                for _ in range(args.ticks_between_frames):
                    world.tick()
                frame_no = world.tick()
                t0 = time.time()
                while latest.get("m") is None or latest["m"].frame != frame_no:
                    if time.time() - t0 > 10:
                        raise TimeoutError("lidar measurement did not arrive")
                    time.sleep(0.002)
                m = latest["m"]
                tf = m.transform  # sensor world pose at this measurement
                sensor_v = sensor_world_velocity(ego, tf) if ego is not None else np.zeros(3)
                raw = np.frombuffer(m.raw_data, dtype=np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("cos", "f4"),
                                                                 ("idx", "u4"), ("tag", "u4")]))
                if ego is not None:
                    raw = raw[raw["idx"] != ego.id]  # never trace the radar's own car
                # lidar frame: x fwd, y right, z up (UE) -> x right, y fwd, z up
                xyz = np.stack([raw["y"], raw["x"], raw["z"]], 1).astype(np.float32)
                vel_table = actor_velocities(world, tf, sensor_v)
                vel = np.zeros_like(xyz)
                idx = raw["idx"].astype(np.int64)
                for aid in np.unique(idx):
                    if int(aid) in vel_table:
                        vel[idx == aid] = vel_table[int(aid)]
                np.savez_compressed(pos_dir / f"frame_{f:03d}.npz", xyz=xyz, cos_inc=raw["cos"].astype(np.float32),
                                    tag=raw["tag"].astype(np.uint8), object_idx=idx.astype(np.int64), velocity=vel)
                (pos_dir / f"frame_{f:03d}.json").write_text(json.dumps({
                    "town": town, "position": p, "frame": f, "carla_frame": int(frame_no), "mount": args.mount,
                    "sensor_transform": {"x": tf.location.x, "y": tf.location.y, "z": tf.location.z,
                                         "yaw": tf.rotation.yaw},
                    "sensor_velocity_world": sensor_v.tolist(),
                    "ego_speed_m_s": float(np.linalg.norm(sensor_v)),
                    "sensor_height_above_lane_m": args.mount_height if ego is not None else None,
                    "ray_solid_angle_sr": solid_angle,
                    "vehicles": vehicle_records(world, tf, sensor_v, skip_ids=(ego.id,) if ego is not None else ())}))
            lidar.stop(); lidar.destroy()
            teardown(client, world, vehicles + ([ego.id] if ego is not None else []), walkers, controllers, args.tm_port)
            (pos_dir / "DONE").write_text("ok")
            print(town, p, "done", flush=True)
        settings.synchronous_mode = False; world.apply_settings(settings)


if __name__ == "__main__":
    main()
