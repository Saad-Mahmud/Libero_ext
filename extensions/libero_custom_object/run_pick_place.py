import argparse
import json
import os
import random
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
    "ball": "ball_1",
    "baseball": "ball_1",
}

OBJECT_GRASP_FRACTIONS = {
    # Lower grasps look more natural for the thin / long TurboSquid meshes.
    "custom_obvious_mesh_turbosquid_dynamite_1": 0.45,
    "custom_obvious_mesh_turbosquid_pistol_1": 0.20,
    "custom_obvious_mesh_turbosquid_bullet_1": 0.25,
    "custom_obvious_mesh_cartoon_bomb_1": 0.55,
    "custom_rc_tomato_ov_tomato_0_1": 0.55,
    "custom_stove_mini_metal_pot": 0.55,
    "custom_stove_mini_saucepan": 0.55,
    "custom_stove_mini_kettle": 0.55,
    "custom_moka_pot_small": 0.50,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Script a Panda pick-and-place motion in a LIBERO custom-object scene "
            "and save the rollout as a video."
        )
    )
    parser.add_argument(
        "object",
        nargs="?",
        default=None,
        help=(
            "Object instance to move. Short aliases work: pistol, bullet, "
            "dynamite, bomb, tomato, ball."
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
        help="Output video path. Defaults to outputs/pick_place_<object>_to_<target>.mp4.",
    )
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=1.0,
        help="Scale fixed render camera x/y position away from the scene origin.",
    )
    parser.add_argument("--camera-offset-x", type=float, default=0.0)
    parser.add_argument("--camera-offset-y", type=float, default=0.0)
    parser.add_argument("--camera-offset-z", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-profile", default="baseline")
    parser.add_argument("--video-pix-fmt", default="yuv420p")
    parser.add_argument("--video-level", default="3.1")
    parser.add_argument("--video-faststart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--initial-settle-steps",
        type=int,
        default=5,
        help="Simulator steps after reset before recording; matches the image renderer default.",
    )
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument(
        "--target-site",
        default="flat_stove_1_burner",
        help="MuJoCo site used as the placement target.",
    )
    parser.add_argument(
        "--target-object",
        default=None,
        help="Object body used as the placement target. Overrides --target-site when set.",
    )
    parser.add_argument(
        "--target-object-offset",
        default="0 0 0",
        help="World xyz offset added to --target-object body position before placing.",
    )
    parser.add_argument(
        "--placement-mode",
        choices=["surface", "center"],
        default="surface",
        help=(
            "surface places the object bottom at the target site; center places "
            "the object center at the target site, which is useful for inside-volume targets."
        ),
    )
    parser.add_argument(
        "--target-offset",
        default="0 0 0",
        help="World xyz offset added to the target site position before placing.",
    )
    parser.add_argument(
        "--approach-from-site-axis",
        choices=["none", "local_x", "local_neg_x", "local_y", "local_neg_y"],
        default="none",
        help=(
            "Optional local target-site axis used to approach the placement target "
            "horizontally before insertion."
        ),
    )
    parser.add_argument(
        "--target-approach-distance",
        type=float,
        default=0.0,
        help="Meters away from the target site to start the final horizontal insertion.",
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
        help="Small z clearance above the target when releasing.",
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
        "--no-stabilize-objects",
        action="store_true",
        help=(
            "Do not lock free-object poses during scripted rollout. By default "
            "objects are stabilized so round meshes cannot roll away before grasp."
        ),
    )
    parser.add_argument(
        "--list-objects",
        action="store_true",
        help="Print available object instance names and exit after reset.",
    )
    parser.add_argument(
        "--reset-attempts",
        type=int,
        default=10,
        help="Retry scene construction/reset this many times if placement randomization fails.",
    )
    return parser.parse_args()


def sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "object"


def output_path_for(args, object_name):
    if args.output:
        return Path(args.output).expanduser().resolve()
    if args.target_object:
        target_label = sanitize_filename(args.target_object)
    elif args.target_site == "flat_stove_1_burner":
        target_label = "stove"
    else:
        target_label = sanitize_filename(args.target_site)
    filename = f"pick_place_{sanitize_filename(object_name)}_to_{target_label}.mp4"
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


def object_shape(base_env, object_name):
    category = base_env.objects_dict[object_name].category_name
    compiled_offsets = compiled_collision_vertical_offsets(base_env, object_name)
    if compiled_offsets is not None:
        return {"bottom_z": compiled_offsets[0], "top_z": compiled_offsets[1], "source": "compiled"}

    manifest_entries = load_manifest_entries()
    if category in manifest_entries:
        bottom_z, top_z = object_vertical_offsets(manifest_entries[category]["xml_path"])
        return {"bottom_z": bottom_z, "top_z": top_z, "source": "manifest"}

    return {"bottom_z": 0.0, "top_z": 0.06, "source": "fallback"}


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
    def __init__(
        self,
        sim,
        output_path,
        camera,
        width,
        height,
        fps,
        record_every,
        codec,
        profile,
        pix_fmt,
        level,
        faststart,
    ):
        self.sim = sim
        self.camera = camera
        self.width = width
        self.height = height
        self.record_every = max(1, record_every)
        self.step_count = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio

        output_params = []
        if profile:
            output_params.extend(["-profile:v", str(profile)])
        if level:
            output_params.extend(["-level", str(level)])
        if faststart:
            output_params.extend(["-movflags", "+faststart"])

        self.writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec=str(codec),
            pixelformat=str(pix_fmt),
            macro_block_size=None,
            output_params=output_params,
        )
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
        frame = np.flipud(frame)
        self.writer.append_data(frame)

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
        self.pose_locks = {} if args.no_stabilize_objects else self.capture_object_pose_locks()

    def eef_pos(self):
        return self.base_env.sim.data.site_xpos[self.robot.eef_site_id].copy()

    def object_pos(self):
        body_id = self.base_env.obj_body_id[self.object_name]
        return self.base_env.sim.data.body_xpos[body_id].copy()

    def object_quat(self):
        joint = self.base_env.objects_dict[self.object_name].joints[-1]
        return self.base_env.sim.data.get_joint_qpos(joint).copy()[3:7]

    def object_joint(self, object_name):
        return self.base_env.objects_dict[object_name].joints[-1]

    def capture_object_pose_locks(self):
        locks = {}
        for name, obj in self.base_env.objects_dict.items():
            joint = obj.joints[-1]
            locks[name] = self.base_env.sim.data.get_joint_qpos(joint).copy()
        return locks

    def update_pose_lock(self, object_name):
        if object_name not in self.pose_locks:
            return
        joint = self.object_joint(object_name)
        self.pose_locks[object_name] = self.base_env.sim.data.get_joint_qpos(joint).copy()

    def zero_joint_velocity(self, joint):
        try:
            qvel_addr = self.base_env.sim.model.get_joint_qvel_addr(joint)
            self.base_env.sim.data.qvel[qvel_addr] = 0.0
        except Exception:
            pass

    def stabilize_object_poses(self):
        if not self.pose_locks:
            return
        for object_name, qpos in self.pose_locks.items():
            if object_name == self.object_name and self.attached:
                continue
            joint = self.object_joint(object_name)
            self.base_env.sim.data.set_joint_qpos(joint, qpos)
            self.zero_joint_velocity(joint)
        self.base_env.sim.forward()

    def site_pos(self, site_name):
        site_id = self.base_env.sim.model.site_name2id(site_name)
        return self.base_env.sim.data.site_xpos[site_id].copy()

    def site_xmat(self, site_name):
        site_id = self.base_env.sim.model.site_name2id(site_name)
        return self.base_env.sim.data.site_xmat[site_id].reshape(3, 3).copy()

    def target_pos(self):
        if self.args.target_object:
            body_id = self.base_env.obj_body_id[self.args.target_object]
            return self.base_env.sim.data.body_xpos[body_id].copy() + parse_vec3(
                self.args.target_object_offset, [0, 0, 0]
            )
        return self.site_pos(self.args.target_site) + parse_vec3(
            self.args.target_offset, [0, 0, 0]
        )

    def target_approach_pos(self, target):
        if (
            self.args.approach_from_site_axis == "none"
            or self.args.target_approach_distance <= 0
        ):
            return target.copy()

        axes = {
            "local_x": np.array([1.0, 0.0, 0.0]),
            "local_neg_x": np.array([-1.0, 0.0, 0.0]),
            "local_y": np.array([0.0, 1.0, 0.0]),
            "local_neg_y": np.array([0.0, -1.0, 0.0]),
        }
        if self.args.target_object:
            axis = axes[self.args.approach_from_site_axis]
        else:
            axis = self.site_xmat(self.args.target_site) @ axes[
                self.args.approach_from_site_axis
            ]
        axis[2] = 0.0
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            return target.copy()
        axis /= norm
        return target + axis * self.args.target_approach_distance

    def set_object_pose(self, pos, quat=None):
        obj = self.base_env.objects_dict[self.object_name]
        joint = obj.joints[-1]
        qpos = self.base_env.sim.data.get_joint_qpos(joint).copy()
        qpos[:3] = np.asarray(pos, dtype=float)
        if quat is not None:
            qpos[3:7] = quat
        self.base_env.sim.data.set_joint_qpos(joint, qpos)
        self.zero_joint_velocity(joint)

        self.base_env.sim.forward()

    def object_bottom_top_world(self):
        pos_z = self.object_pos()[2]
        return pos_z + self.shape["bottom_z"], pos_z + self.shape["top_z"]

    def step(self, action):
        self.env.step(action)
        if self.attached:
            self.set_object_pose(self.eef_pos() + self.attach_offset, self.attach_quat)
        self.stabilize_object_poses()
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

    def target_body_z(self, target):
        if self.args.placement_mode == "center":
            object_center_z = (self.shape["bottom_z"] + self.shape["top_z"]) / 2.0
            return target[2] - object_center_z + self.args.place_clearance
        return target[2] - self.shape["bottom_z"] + self.args.place_clearance

    def release_on_target(self, target):
        body_z = self.target_body_z(target)
        self.set_object_pose([target[0], target[1], body_z], self.attach_quat)
        self.update_pose_lock(self.object_name)
        self.attached = False

    def run(self):
        obj_pos = self.object_pos()
        bottom_z, top_z = self.object_bottom_top_world()
        height = max(0.02, top_z - bottom_z)
        grasp_fraction = self.args.grasp_fraction
        if grasp_fraction is None:
            category = self.base_env.objects_dict[self.object_name].category_name
            grasp_fraction = OBJECT_GRASP_FRACTIONS.get(
                self.object_name,
                OBJECT_GRASP_FRACTIONS.get(category, 0.35),
            )
        grasp_z = bottom_z + np.clip(grasp_fraction, 0.05, 0.95) * height
        approach_z = max(self.args.transport_z, top_z + self.args.approach_clearance)
        object_xy = obj_pos[:2]
        target = self.target_pos()
        approach_target = self.target_approach_pos(target)

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
        self.servo_to(
            [approach_target[0], approach_target[1], self.args.transport_z],
            gripper=1.0,
        )

        release_body_z = self.target_body_z(target)
        release_eef_z = (
            release_body_z - self.attach_offset[2] if self.attached else target[2] + 0.05
        )
        self.servo_to(
            [approach_target[0], approach_target[1], release_eef_z],
            gripper=1.0,
        )
        self.servo_to([target[0], target[1], release_eef_z], gripper=1.0)

        if self.attached:
            self.release_on_target(target)

        self.hold(gripper=-1.0, steps=35)
        self.servo_to(
            [approach_target[0], approach_target[1], self.args.transport_z],
            gripper=-1.0,
        )
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


def make_reset_env(args):
    last_error = None
    for attempt in range(1, args.reset_attempts + 1):
        env = None
        try:
            env = make_env(args)
            env.reset()
            if attempt > 1:
                print(f"reset_attempts_used: {attempt}")
            return env
        except Exception as exc:
            last_error = exc
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if "Cannot place all objects" not in str(exc):
                raise
            print(f"reset_retry: {attempt}/{args.reset_attempts}")
    raise last_error


def apply_render_camera_adjustment(sim, args):
    if (
        args.camera_distance_scale == 1.0
        and args.camera_offset_x == 0.0
        and args.camera_offset_y == 0.0
        and args.camera_offset_z == 0.0
    ):
        return
    camera_id = sim.model.camera_name2id(args.camera)
    sim.model.cam_pos[camera_id][:2] *= args.camera_distance_scale
    sim.model.cam_pos[camera_id][0] += args.camera_offset_x
    sim.model.cam_pos[camera_id][1] += args.camera_offset_y
    sim.model.cam_pos[camera_id][2] += args.camera_offset_z
    sim.forward()


def settle_initial_state(env, steps):
    for _ in range(max(0, int(steps))):
        env.step([0.0] * 7)


def object_inside_site(base_env, object_name, site_name):
    site_id = base_env.sim.model.site_name2id(site_name)
    site_pos = base_env.sim.data.site_xpos[site_id]
    site_mat = base_env.sim.data.site_xmat[site_id].reshape(3, 3)
    site_size = base_env.sim.model.site_size[site_id]
    object_pos = base_env.sim.data.body_xpos[base_env.obj_body_id[object_name]]

    total_size = np.abs(site_mat @ site_size)
    lower = site_pos - total_size
    upper = site_pos + total_size
    lower[2] -= 0.01
    return bool(np.all(object_pos > lower) and np.all(object_pos < upper))


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    if not Path(args.bddl).expanduser().exists():
        raise FileNotFoundError(f"BDDL file does not exist: {args.bddl}")

    env = make_reset_env(args)
    try:
        apply_render_camera_adjustment(env.env.sim, args)
        settle_initial_state(env, args.initial_settle_steps)
        object_names = sorted(env.env.objects_dict.keys())
        if args.list_objects:
            print("available_objects:")
            for name in object_names:
                print(f"  {name}")
            return
        if args.object is None:
            raise ValueError(
                "Missing object name. Use one of: pistol, bullet, dynamite, bomb, tomato, ball; "
                "or pass --list-objects."
            )

        object_name = resolve_object_name(args.object, object_names)
        category = env.env.objects_dict[object_name].category_name
        shape = object_shape(env.env, object_name)

        output_path = output_path_for(args, object_name)
        recorder = VideoRecorder(
            env.env.sim,
            output_path,
            args.camera,
            args.width,
            args.height,
            args.fps,
            args.record_every,
            args.video_codec,
            args.video_profile,
            args.video_pix_fmt,
            args.video_level,
            args.video_faststart,
        )
        try:
            controller = PickPlaceController(env, object_name, shape, args, recorder)
            controller.run()
        finally:
            recorder.close()

        final_pos = env.env.sim.data.body_xpos[env.env.obj_body_id[object_name]].copy()
        target_pos = controller.target_pos()
        target_distance = float(np.linalg.norm(final_pos[:2] - target_pos[:2]))
        inside_target = (
            None
            if args.target_object
            else object_inside_site(env.env, object_name, args.target_site)
        )
        result = {
            "moved_object": object_name,
            "moved_category": category,
            "shape_source": shape["source"],
            "target_site": args.target_site,
            "target_object": args.target_object,
            "target_position": target_pos.tolist(),
            "final_position": final_pos.tolist(),
            "target_xy_distance": target_distance,
            "inside_target_site": inside_target,
            "saved_video": str(output_path),
            "mode": "physics_only" if args.physics_only else "assisted_pick_place",
            "object_pose_stabilization": not args.no_stabilize_objects,
        }
        print(f"moved_object: {object_name}")
        print(f"moved_category: {category}")
        print(f"shape_source: {shape['source']}")
        print(f"final_position: {final_pos.tolist()}")
        print(f"target_xy_distance: {target_distance}")
        print(f"inside_target_site: {inside_target}")
        print(f"saved_video: {output_path}")
        if not args.physics_only:
            print("mode: assisted_pick_place")
        else:
            print("mode: physics_only")
        print(f"object_pose_stabilization: {not args.no_stabilize_objects}")
        print("result_json:", json.dumps(result, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
