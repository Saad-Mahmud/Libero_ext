import argparse
import os
import time
from pathlib import Path

from libero_custom_object import (
    get_can_opener_bddl_path,
    get_four_objects_bddl_path,
    get_hammer_bddl_path,
    get_hazard_tools_bddl_path,
    get_kitchen_hazards_bddl_path,
    get_knife_bddl_path,
    get_sample_bddl_path,
    get_scissors_bddl_path,
    get_steak_knife_bddl_path,
    get_two_object_bddl_path,
    register_custom_objects,
)


BDDL_ALIASES = {
    "sample": get_sample_bddl_path,
    "one_object": get_sample_bddl_path,
    "two_object": get_two_object_bddl_path,
    "knife": get_knife_bddl_path,
    "scissors": get_scissors_bddl_path,
    "hammer": get_hammer_bddl_path,
    "hazards": get_hazard_tools_bddl_path,
    "dangerous_tools": get_hazard_tools_bddl_path,
    "steak_knife": get_steak_knife_bddl_path,
    "steak": get_steak_knife_bddl_path,
    "can_opener": get_can_opener_bddl_path,
    "kitchen_hazards": get_kitchen_hazards_bddl_path,
    "four_objects": get_four_objects_bddl_path,
    "mixed_hazards": get_four_objects_bddl_path,
}


def resolve_bddl(value):
    if value in BDDL_ALIASES:
        return BDDL_ALIASES[value]()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a LIBERO custom-object BDDL config."
    )
    parser.add_argument(
        "bddl",
        help=(
            "Path to a .bddl file, or one of: sample, one_object, two_object, "
            "knife, scissors, hammer, hazards, steak_knife, can_opener, "
            "kitchen_hazards, four_objects"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["viewer", "image", "check"],
        default="viewer",
        help="viewer opens the live simulator, image saves a PNG, check only validates reset.",
    )
    parser.add_argument("--camera", default=None)
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "outputs" / "scene.png"),
        help="Output PNG path for --mode image.",
    )
    return parser.parse_args()


def get_problem_info(bddl_file):
    import libero.libero.envs.bddl_utils as BDDLUtils

    return BDDLUtils.get_problem_info(bddl_file)


def make_viewer_env(args, bddl_file, problem_info):
    from libero.libero.envs import TASK_MAPPING
    from robosuite import load_controller_config

    return TASK_MAPPING[problem_info["problem_name"]](
        bddl_file_name=bddl_file,
        robots=["Panda"],
        controller_configs=load_controller_config(default_controller=args.controller),
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera or "frontview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=args.control_freq,
    )


def viewer_window_is_open(env):
    viewer = getattr(env, "viewer", None)
    if viewer is None:
        return False

    if viewer.__class__.__name__ == "OpenCVRenderer":
        try:
            import cv2

            return cv2.getWindowProperty("offscreen render", cv2.WND_PROP_VISIBLE) >= 1
        except Exception:
            return False

    window = getattr(getattr(viewer, "viewer", viewer), "window", None)
    if window is not None:
        try:
            import glfw

            return not glfw.window_should_close(window)
        except Exception:
            return True

    return True


def make_offscreen_env(args, bddl_file, use_camera_obs):
    from libero.libero.envs import OffScreenRenderEnv

    camera = args.camera or "agentview"
    return OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        robots=["Panda"],
        use_camera_obs=use_camera_obs,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=[camera],
        camera_heights=args.height,
        camera_widths=args.width,
    )


def run_viewer(args, bddl_file, problem_info):
    env = make_viewer_env(args, bddl_file, problem_info)
    env.reset()
    print("viewer_started:", problem_info["language_instruction"])
    print("bddl_file:", bddl_file)
    print("close the viewer window, press q/Esc in the viewer, or press Ctrl+C here")

    action = [0.0] * 7
    stop_requested = {"value": False}
    if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "add_keypress_callback"):
        def request_stop(key):
            if key in (ord("q"), ord("Q"), 27):
                stop_requested["value"] = True

        env.viewer.add_keypress_callback(request_stop)

    try:
        while True:
            env.step(action)
            env.render()
            if stop_requested["value"] or not viewer_window_is_open(env):
                print("viewer_closed")
                break
            time.sleep(1.0 / args.control_freq)
    except KeyboardInterrupt:
        print("viewer_closed")
    finally:
        env.close()


def run_image(args, bddl_file, problem_info):
    import imageio.v2 as imageio
    import numpy as np

    env = make_offscreen_env(args, bddl_file, use_camera_obs=True)
    obs = env.reset()
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)

    camera = args.camera or "agentview"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output, np.flipud(obs[f"{camera}_image"]))
    print("saved_render:", output)
    print("sample_language:", problem_info["language_instruction"])
    env.close()


def run_check(args, bddl_file, problem_info):
    env = make_offscreen_env(args, bddl_file, use_camera_obs=False)
    env.reset()
    print("env_reset_ok:", sorted(env.env.objects_dict.keys()))
    print("sample_language:", problem_info["language_instruction"])
    env.close()


def main():
    args = parse_args()

    if args.mode == "viewer":
        os.environ.setdefault("MUJOCO_GL", "glfw")
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")

    register_custom_objects()
    bddl_file = resolve_bddl(args.bddl)
    if not os.path.exists(bddl_file):
        raise FileNotFoundError(f"BDDL file does not exist: {bddl_file}")

    problem_info = get_problem_info(bddl_file)
    if args.mode == "viewer":
        run_viewer(args, bddl_file, problem_info)
    elif args.mode == "image":
        run_image(args, bddl_file, problem_info)
    else:
        run_check(args, bddl_file, problem_info)


if __name__ == "__main__":
    main()
