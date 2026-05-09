import argparse
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml


os.environ.setdefault("MUJOCO_GL", "egl")


EXTENSION_ROOT = Path(__file__).resolve().parent
DEFAULT_BDDL = (
    EXTENSION_ROOT
    / "libero_custom_object"
    / "bddl_files"
    / "check_stove_turbosquid_dynamite_bomb_tomato.bddl"
)
DEFAULT_OUTPUT_DIR = EXTENSION_ROOT / "outputs"

OBJECT_ALIASES = {
    "dynamite": "custom_obvious_mesh_turbosquid_dynamite_1",
    "ts_dynamite": "custom_obvious_mesh_turbosquid_dynamite_1",
    "bomb": "custom_obvious_mesh_cartoon_bomb_1",
    "black_bomb": "custom_obvious_mesh_cartoon_bomb_1",
    "cartoon_bomb": "custom_obvious_mesh_cartoon_bomb_1",
    "pistol": "custom_obvious_mesh_turbosquid_pistol_1",
    "gun": "custom_obvious_mesh_turbosquid_pistol_1",
    "bullet": "custom_obvious_mesh_turbosquid_bullet_1",
    "tomato": "custom_rc_tomato_ov_tomato_0_1",
}

OBJECT_GRASP_FRACTIONS = {
    # Lower grasps look more natural for the thin / long TurboSquid meshes.
    "custom_obvious_mesh_turbosquid_dynamite_1": 0.45,
    "custom_obvious_mesh_turbosquid_pistol_1": 0.20,
    "custom_obvious_mesh_turbosquid_bullet_1": 0.25,
    "custom_obvious_mesh_cartoon_bomb_1": 0.55,
    "custom_rc_tomato_ov_tomato_0_1": 0.55,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Script a Panda pick-and-place motion in the TurboSquid stove scene "
            "and save the rollout as a video."
        )
    )
    parser.add_argument(
        "object",
        nargs="?",
        default=None,
        help=(
            "Object instance to move. Short aliases work: pistol, bullet, "
            "dynamite, bomb, tomato."
        ),
    )
    parser.add_argument(
        "--bddl",
        default=str(DEFAULT_BDDL),
        help="BDDL file to load. Defaults to the TurboSquid stove scene.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path. Defaults to outputs/pick_place_<object>_to_stove.mp4.",
    )
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument(
        "--target-site",
        default="flat_stove_1_burner",
        help="MuJoCo site used as the stove placement target.",
    )
    parser.add_argument(
        "--transport-z",
        type=float,
        default=1.12,
        help="World z used for carrying the object across the table.",
    )
    parser.add_argument(
        "--approach-clearance",
        type=float,
        default=0.10,
        help="Meters above object top before descending to grasp.",
    )
    parser.add_argument(
        "--grasp-fraction",
        type=float,
        default=None,
        help=(
            "Height fraction from object bottom to top used for the gripper center. "
            "Defaults are object-specific."
        ),
    )
    parser.add_argument(
        "--pregrasp-lift",
        type=float,
        default=0.015,
        help="Meters above the grasp point used for the final pre-grasp waypoint.",
    )
    parser.add_argument(
        "--place-clearance",
        type=float,
        default=0.003,
        help="Small z clearance above the stove target surface when releasing.",
    )
    parser.add_argument("--pos-gain", type=float, default=8.0)
    parser.add_argument("--pos-tol", type=float, default=0.008)
    parser.add_argument(
        "--max-servo-steps",
        type=int,
        default=140,
        help="Maximum controller steps per waypoint.",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=1,
        help="Record every N simulator steps.",
    )
    parser.add_argument(
        "--physics-only",
        action="store_true",
        help=(
            "Disable object attachment assistance after grasp. This is more "
            "physical, but thin meshes can slip from the gripper."
        ),
    )
    parser.add_argument(
        "--list-objects",
        action="store_true",
        help="Print available object instance names and exit after reset.",
    )
    return parser.parse_args()


def sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "object"


def output_path_for(args, object_name):
    if args.output:
        return Path(args.output).expanduser().resolve()
    filename = f"pick_place_{sanitize_filename(object_name)}_to_stove.mp4"
    return (DEFAULT_OUTPUT_DIR / filename).resolve()


def load_manifest_entries():
    from libero_custom_object import get_manifest_path

    manifest_path = Path(get_manifest_path()).resolve()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    assets_root = manifest_path.parent
    entries = {}
    for entry in manifest.get("objects", []):
        entries[entry["category_name"]] = {
            **entry,
            "xml_path": assets_root / "objects" / entry["asset_name"] / entry["xml"],
        }
    return entries


def parse_vec3(value, default):
    if value is None:
        return np.array(default, dtype=float)
    parts = [float(part) for part in value.split()]
    if len(parts) != 3:
        return np.array(default, dtype=float)
    return np.array(parts, dtype=float)


def object_vertical_offsets(xml_path):
    """Return bottom/top z offsets relative to the object root body."""
    root = ET.parse(xml_path).getroot()

    bottom_site = root.find(".//site[@name='bottom_site']")
    top_site = root.find(".//site[@name='top_site']")
    if bottom_site is not None and top_site is not None:
        bottom = parse_vec3(bottom_site.get("pos"), [0, 0, 0])[2]
        top = parse_vec3(top_site.get("pos"), [0, 0, 0.06])[2]
        return min(bottom, top), max(bottom, top)

    bbox = root.find(".//geom[@name='reg_bbox']")
    if bbox is not None:
        pos = parse_vec3(bbox.get("pos"), [0, 0, 0])
        size = parse_vec3(bbox.get("size"), [0.03, 0.03, 0.03])
        return pos[2] - size[2], pos[2] + size[2]

    return 0.0, 0.06


def compiled_collision_vertical_offsets(base_env, object_name):
    """Return bottom/top z offsets from compiled collision geoms when available."""
    body_z = base_env.sim.data.body_xpos[base_env.obj_body_id[object_name]][2]
    z_min = None
    z_max = None

    for geom_id in range(base_env.sim.model.ngeom):
        name = base_env.sim.model.geom_id2name(geom_id) or ""
        if not name.startswith(f"{object_name}_"):
            continue
        if base_env.sim.model.geom_group[geom_id] != 0:
            continue
        if base_env.sim.model.geom_contype[geom_id] == 0:
            continue

        geom_type = int(base_env.sim.model.geom_type[geom_id])
        geom_pos = base_env.sim.data.geom_xpos[geom_id]
        geom_size = base_env.sim.model.geom_size[geom_id]
        geom_xmat = base_env.sim.data.geom_xmat[geom_id].reshape(3, 3)

        if geom_type == 6:  # box
            half_extents = np.abs(geom_xmat) @ geom_size[:3]
            geom_min = geom_pos[2] - half_extents[2]
            geom_max = geom_pos[2] + half_extents[2]
        else:
            radius = float(np.max(geom_size[:3]))
            geom_min = geom_pos[2] - radius
            geom_max = geom_pos[2] + radius

        z_min = geom_min if z_min is None else min(z_min, geom_min)
        z_max = geom_max if z_max is None else max(z_max, geom_max)

    if z_min is None or z_max is None:
        return None
    return z_min - body_z, z_max - body_z


def resolve_object_name(requested, object_names):
    if requested in object_names:
        return requested
    if requested in OBJECT_ALIASES and OBJECT_ALIASES[requested] in object_names:
        return OBJECT_ALIASES[requested]

    matches = [name for name in object_names if requested.lower() in name.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"Unknown object '{requested}'. Available objects: {', '.join(sorted(object_names))}"
        )
    raise ValueError(
        f"Object name '{requested}' is ambiguous. Matches: {', '.join(sorted(matches))}"
    )


class VideoRecorder:
    def __init__(self, sim, output_path, camera, width, height, fps, record_every):
        import imageio.v2 as imageio

        self.sim = sim
        self.camera = camera
        self.width = width
        self.height = height
        self.record_every = max(1, record_every)
        self.step_count = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(output_path), fps=fps)
        self.output_path = output_path

    def append(self, force=False):
        self.step_count += 1
        if not force and self.step_count % self.record_every != 0:
            return
        frame = self.sim.render(
            camera_name=self.camera,
            width=self.width,
            height=self.height,
            depth=False,
        )
        self.writer.append_data(np.flipud(frame))

    def close(self):
        self.writer.close()


class PickPlaceController:
    def __init__(self, env, object_name, shape, args, recorder):
        self.env = env
        self.base_env = env.env
        self.object_name = object_name
        self.shape = shape
        self.args = args
        self.recorder = recorder
        self.robot = self.base_env.robots[0]
        self.attached = False
        self.attach_offset = np.zeros(3)
        self.attach_quat = None

    def eef_pos(self):
        return self.base_env.sim.data.site_xpos[self.robot.eef_site_id].copy()

    def object_pos(self):
        body_id = self.base_env.obj_body_id[self.object_name]
        return self.base_env.sim.data.body_xpos[body_id].copy()

    def object_quat(self):
        joint = self.base_env.objects_dict[self.object_name].joints[-1]
        return self.base_env.sim.data.get_joint_qpos(joint).copy()[3:7]

    def site_pos(self, site_name):
        site_id = self.base_env.sim.model.site_name2id(site_name)
        return self.base_env.sim.data.site_xpos[site_id].copy()

    def set_object_pose(self, pos, quat=None):
        obj = self.base_env.objects_dict[self.object_name]
        joint = obj.joints[-1]
        qpos = self.base_env.sim.data.get_joint_qpos(joint).copy()
        qpos[:3] = np.asarray(pos, dtype=float)
        if quat is not None:
            qpos[3:7] = quat
        self.base_env.sim.data.set_joint_qpos(joint, qpos)

        try:
            qvel_addr = self.base_env.sim.model.get_joint_qvel_addr(joint)
            self.base_env.sim.data.qvel[qvel_addr] = 0.0
        except Exception:
            pass

        self.base_env.sim.forward()

    def object_bottom_top_world(self):
        pos_z = self.object_pos()[2]
        return pos_z + self.shape["bottom_z"], pos_z + self.shape["top_z"]

    def step(self, action):
        self.env.step(action)
        if self.attached:
            self.set_object_pose(self.eef_pos() + self.attach_offset, self.attach_quat)
        self.recorder.append()

    def hold(self, gripper, steps):
        action = np.zeros(7)
        action[-1] = gripper
        for _ in range(steps):
            self.step(action)

    def servo_to(self, target, gripper, max_steps=None, tolerance=None):
        max_steps = max_steps or self.args.max_servo_steps
        tolerance = tolerance or self.args.pos_tol
        target = np.asarray(target, dtype=float)

        for i in range(max_steps):
            delta = target - self.eef_pos()
            action = np.zeros(7)
            action[:3] = np.clip(delta * self.args.pos_gain, -1.0, 1.0)
            action[-1] = gripper
            self.step(action)
            if i > 5 and np.linalg.norm(delta) < tolerance:
                break

    def attach_object(self):
        self.attach_offset = self.object_pos() - self.eef_pos()
        self.attach_quat = self.object_quat()
        self.attached = True

    def release_on_stove(self, target_site):
        target = self.site_pos(target_site)
        body_z = target[2] - self.shape["bottom_z"] + self.args.place_clearance
        self.set_object_pose([target[0], target[1], body_z], self.attach_quat)
        self.attached = False

    def run(self):
        obj_pos = self.object_pos()
        bottom_z, top_z = self.object_bottom_top_world()
        height = max(0.02, top_z - bottom_z)
        grasp_fraction = self.args.grasp_fraction
        if grasp_fraction is None:
            grasp_fraction = OBJECT_GRASP_FRACTIONS.get(self.object_name, 0.35)
        grasp_z = bottom_z + np.clip(grasp_fraction, 0.05, 0.95) * height
        approach_z = max(self.args.transport_z, top_z + self.args.approach_clearance)
        object_xy = obj_pos[:2]
        target = self.site_pos(self.args.target_site)

        self.recorder.append(force=True)
        self.hold(gripper=-1.0, steps=20)
        self.servo_to([object_xy[0], object_xy[1], approach_z], gripper=-1.0)
        self.servo_to(
            [object_xy[0], object_xy[1], grasp_z + self.args.pregrasp_lift],
            gripper=-1.0,
        )
        self.servo_to([object_xy[0], object_xy[1], grasp_z], gripper=-1.0)
        self.hold(gripper=1.0, steps=45)

        if not self.args.physics_only:
            self.attach_object()

        self.servo_to([object_xy[0], object_xy[1], self.args.transport_z], gripper=1.0)
        self.servo_to([target[0], target[1], self.args.transport_z], gripper=1.0)

        release_body_z = target[2] - self.shape["bottom_z"] + self.args.place_clearance
        release_eef_z = release_body_z - self.attach_offset[2] if self.attached else target[2] + 0.05
        self.servo_to([target[0], target[1], release_eef_z], gripper=1.0)

        if self.attached:
            self.release_on_stove(self.args.target_site)

        self.hold(gripper=-1.0, steps=35)
        self.servo_to([target[0], target[1], self.args.transport_z], gripper=-1.0)
        self.hold(gripper=-1.0, steps=20)
        self.recorder.append(force=True)


def make_env(args):
    from libero.libero.envs import OffScreenRenderEnv
    from libero_custom_object import register_custom_objects

    register_custom_objects()
    return OffScreenRenderEnv(
        bddl_file_name=str(Path(args.bddl).expanduser().resolve()),
        robots=["Panda"],
        controller=args.controller,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=[args.camera],
        camera_heights=args.height,
        camera_widths=args.width,
        control_freq=args.control_freq,
        horizon=args.horizon,
        ignore_done=True,
    )


def main():
    args = parse_args()
    if not Path(args.bddl).expanduser().exists():
        raise FileNotFoundError(f"BDDL file does not exist: {args.bddl}")

    env = make_env(args)
    try:
        env.reset()
        object_names = sorted(env.env.objects_dict.keys())
        if args.list_objects:
            print("available_objects:")
            for name in object_names:
                print(f"  {name}")
            return
        if args.object is None:
            raise ValueError(
                "Missing object name. Use one of: pistol, bullet, dynamite, bomb, tomato; "
                "or pass --list-objects."
            )

        object_name = resolve_object_name(args.object, object_names)
        category = env.env.objects_dict[object_name].category_name
        manifest_entries = load_manifest_entries()
        if category not in manifest_entries:
            raise KeyError(f"No manifest entry found for category '{category}'")

        xml_path = manifest_entries[category]["xml_path"]
        compiled_offsets = compiled_collision_vertical_offsets(env.env, object_name)
        if compiled_offsets is None:
            bottom_z, top_z = object_vertical_offsets(xml_path)
        else:
            bottom_z, top_z = compiled_offsets
        shape = {"bottom_z": bottom_z, "top_z": top_z}

        output_path = output_path_for(args, object_name)
        recorder = VideoRecorder(
            env.env.sim,
            output_path,
            args.camera,
            args.width,
            args.height,
            args.fps,
            args.record_every,
        )
        try:
            controller = PickPlaceController(env, object_name, shape, args, recorder)
            controller.run()
        finally:
            recorder.close()

        final_pos = env.env.sim.data.body_xpos[env.env.obj_body_id[object_name]].copy()
        print(f"moved_object: {object_name}")
        print(f"final_position: {final_pos.tolist()}")
        print(f"saved_video: {output_path}")
        if not args.physics_only:
            print("mode: assisted_pick_place")
        else:
            print("mode: physics_only")
    finally:
        env.close()


if __name__ == "__main__":
    main()
