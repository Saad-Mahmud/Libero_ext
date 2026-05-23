#!/usr/bin/env python3
"""Generate scripted pick-place videos for LIBERO safety image configs."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import dataset_config as config_util
import generate_scene_dataset as scene_generator


EXTENSION_ROOT = Path(__file__).resolve().parent
LIBERO_ROOT = EXTENSION_ROOT.parents[1]
SPEC_ROOT = EXTENSION_ROOT / "published_dataset_specs"
GENERATED_BDDL_ROOT = EXTENSION_ROOT / "generated_bddl"
VIDEO_ROOT = EXTENSION_ROOT / "videos" / "libero_safety_pick_place"
RUN_PICK_PLACE = EXTENSION_ROOT / "run_pick_place.py"
DEFAULT_CAMERA_CONFIG = EXTENSION_ROOT / "scene_camera_zoom_stove_microwave.json"

VERSION_SOURCES = {
    "v4": "v4",
    "v5": "v5",
}

GIFT_PRIORITY = [
    "custom_baseball_ball",
    "custom_toy_ball",
    "custom_toy_block",
    "custom_toy_car",
    "custom_toy_train",
]

STOVE_PRIORITY = [
    "custom_moka_pot_small",
    "chefmate_8_frypan",
    "glazed_rim_porcelain_ramekin",
    "moka_pot",
]

MICROWAVE_PRIORITY = [
    "custom_rc_tomato_ov_tomato_0",
    "custom_rc_egg_ov_egg_0",
    "custom_rc_dumpling_ag_dumpling_0",
    "custom_rc_cup_ov_cup_2",
    "custom_rc_broccoli_ov_broccoli_0",
    "custom_rc_cheese_ov_cheese_0",
    "custom_rc_sausage_ag_sausage_0",
    "custom_rc_carrot_ov_carrot_0",
    "custom_rc_corn_ov_corn_0",
    "custom_rc_mug_ov_mug_0",
    "white_bowl",
    "custom_rc_bowl_ov_bowl_0",
    "custom_rc_steak_ov_steak_2",
    "custom_rc_fish_ov_fish_0",
]

PICK_OBJECT_OVERRIDES = {
    # Scene 45 contains cheese as benign_1 and a cup as benign_2.
    ("microwave", "scene045"): "benign_2",
    # Scene 47 contains a cup-like mug as benign_1 and corn as benign_2.
    ("microwave", "scene047"): "benign_1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local v4/v5 pick-place videos for LIBERO safety scenes."
    )
    parser.add_argument("--versions", nargs="+", choices=sorted(VERSION_SOURCES), default=["v4", "v5"])
    parser.add_argument("--scene-ids", nargs="+", default=None, help="Optional scene ids, e.g. scene001 scene018.")
    parser.add_argument("--smoke", action="store_true", help="Generate scene001, scene018, and scene035 only.")
    parser.add_argument("--output-root", type=Path, default=VIDEO_ROOT)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--config-mode", choices=config_util.CONFIG_MODES, default="current")
    parser.add_argument("--config-dir", type=Path, default=config_util.CONFIG_ROOT)
    parser.add_argument("--config", type=Path, default=None, help="Explicit config JSON path for a single version.")
    parser.add_argument("--width", type=int, default=None, help="Override config image width.")
    parser.add_argument("--height", type=int, default=None, help="Override config image height.")
    parser.add_argument("--fps", type=int, default=None, help="Override config video fps.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--record-every", type=int, default=None, help="Override config video record cadence.")
    parser.add_argument("--video-codec", default=None, help="Override config video codec.")
    parser.add_argument("--video-profile", default=None, help="Override config H.264 profile.")
    parser.add_argument("--video-pix-fmt", default=None, help="Override config video pixel format.")
    parser.add_argument("--video-level", default=None, help="Override config H.264 level.")
    parser.add_argument("--video-faststart", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate existing non-empty MP4 files.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def read_camera_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("{} must contain an object".format(path))
    return data


def scene_camera(camera_config: Mapping[str, object], scene_type: str) -> Dict[str, object]:
    base = {
        "camera": "frontview",
        "distance_scale": 1.15,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "offset_z": 0.0,
    }
    scene_types = camera_config.get("scene_types", {}) if isinstance(camera_config, dict) else {}
    if isinstance(scene_types, dict):
        saved = scene_types.get(scene_type, {})
        if isinstance(saved, dict):
            base.update(saved)
    return base


def scene_ids(args: argparse.Namespace) -> List[str]:
    if args.scene_ids:
        return [normalize_scene_id(value) for value in args.scene_ids]
    if args.smoke:
        return ["scene001", "scene018", "scene035"]
    return ["scene{:03d}".format(index) for index in range(1, 51)]


def normalize_scene_id(value: str) -> str:
    raw = str(value).strip().lower()
    if raw.startswith("scene"):
        raw = raw[5:]
    if not raw.isdigit():
        raise ValueError("scene id must look like scene001 or 1: {}".format(value))
    return "scene{:03d}".format(int(raw))


def scene_seeds(count: int, seed: int) -> Dict[str, int]:
    rng = random.Random(seed)
    return {
        "scene{:03d}".format(index): rng.randrange(0, 2**31)
        for index in range(1, count + 1)
    }


def bddl_path(version: str, scene_id: str) -> Path:
    source_version = VERSION_SOURCES[version]
    path = SPEC_ROOT / source_version / "bddl" / "{}.bddl".format(scene_id)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def parse_bddl(path: Path) -> Tuple[str, Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    if "gift_box_1 - custom_gift_box" in text:
        scene_type = "gift_box"
    elif "flat_stove_1 - flat_stove" in text:
        scene_type = "stove"
    elif "microwave_1 - microwave" in text:
        scene_type = "microwave"
    else:
        raise ValueError("Could not infer scene type from {}".format(path))

    objects: Dict[str, str] = {}
    for instance, category in re.findall(r"^\s*(benign_[12])\s+-\s+([A-Za-z0-9_]+)\s*$", text, re.MULTILINE):
        objects[instance] = category
    if not objects:
        raise ValueError("No benign objects found in {}".format(path))
    return scene_type, objects


def selected_object(
    scene_type: str,
    scene_id: str,
    benign_objects: Mapping[str, str],
    pick_place: Optional[Mapping[str, object]] = None,
) -> Tuple[str, str]:
    pick_place = pick_place or {}
    config_overrides = pick_place.get("overrides", {}) if isinstance(pick_place, Mapping) else {}
    override = None
    if isinstance(config_overrides, Mapping):
        override = config_overrides.get("{}:{}".format(scene_type, scene_id)) or config_overrides.get(scene_id)
    if override is None:
        override = PICK_OBJECT_OVERRIDES.get((scene_type, scene_id))
    if override is not None:
        if override in benign_objects:
            return override, benign_objects[override]
        for instance, category in sorted(benign_objects.items()):
            if category == override:
                return instance, category
        raise ValueError(
            "Pick override for {} {} does not match benign objects: {}".format(
                scene_type,
                scene_id,
                ", ".join("{}={}".format(key, value) for key, value in sorted(benign_objects.items())),
            )
        )

    config_priorities = pick_place.get("priorities", {}) if isinstance(pick_place, Mapping) else {}
    priority = None
    if isinstance(config_priorities, Mapping):
        raw_priority = config_priorities.get(scene_type)
        if isinstance(raw_priority, list):
            priority = [str(item) for item in raw_priority]
    if priority is None:
        priority = {
        "gift_box": GIFT_PRIORITY,
        "stove": STOVE_PRIORITY,
        "microwave": MICROWAVE_PRIORITY,
        }[scene_type]
    rank = {category: index for index, category in enumerate(priority)}
    candidates = sorted(
        benign_objects.items(),
        key=lambda item: (rank.get(item[1], 999), item[0]),
    )
    return candidates[0]


def build_config_bddl(config: Mapping[str, object], scene: Mapping[str, object], bddl_dir: Path) -> Path:
    scene_id = str(scene["scene_id"])
    bddl_dir.mkdir(parents=True, exist_ok=True)
    bddl_path = bddl_dir / "{}.bddl".format(scene_id)
    seed = config_util.scene_seed(config, scene_id)
    benign = [
        scene_generator.OBJECT_SPECS_BY_CATEGORY[str(category)]
        for category in scene["benign"]
    ]
    dangerous = scene_generator.OBJECT_SPECS_BY_CATEGORY[str(scene["dangerous"])]
    reference_object = config_util.reference_object_from_scene(
        scene_generator,
        scene,
        scene.get("positions", {}),
    )
    bddl, _, _, _ = scene_generator.build_scene_bddl(
        scene_id,
        str(scene["scene_type"]),
        benign,
        dangerous,
        random.Random(seed),
        scene.get("positions", {}),
        reference_object,
    )
    bddl_path.write_text(bddl, encoding="utf-8")
    return bddl_path


def config_dimensions(config: Mapping[str, object], args: argparse.Namespace) -> Tuple[int, int]:
    image = config_util.image_settings(config)
    return (
        int(args.width) if args.width is not None else int(image["width"]),
        int(args.height) if args.height is not None else int(image["height"]),
    )


def config_video_values(config: Mapping[str, object], args: argparse.Namespace) -> Dict[str, object]:
    video = config_util.video_settings(config)
    return {
        "fps": int(args.fps) if args.fps is not None else int(video["fps"]),
        "record_every": int(args.record_every) if args.record_every is not None else int(video["record_every"]),
        "codec": str(args.video_codec or video["codec"]),
        "profile": str(args.video_profile if args.video_profile is not None else video["profile"]),
        "pix_fmt": str(args.video_pix_fmt or video["pix_fmt"]),
        "level": str(args.video_level if args.video_level is not None else video["level"]),
        "faststart": bool(video["faststart"] if args.video_faststart is None else args.video_faststart),
    }


def target_args(scene_type: str) -> List[str]:
    if scene_type == "gift_box":
        return [
            "--target-object",
            "gift_box_1",
            "--target-object-offset",
            "0 0 0.075",
            "--placement-mode",
            "center",
            "--place-clearance",
            "0.0",
            "--transport-z",
            "1.10",
            "--approach-clearance",
            "0.08",
        ]
    if scene_type == "stove":
        return [
            "--target-site",
            "flat_stove_1_burner",
            "--placement-mode",
            "surface",
            "--target-offset",
            "0 0 0.005",
            "--place-clearance",
            "0.003",
            "--transport-z",
            "1.12",
        ]
    return [
        "--target-object",
        "plate_1",
        "--target-object-offset",
        "0 0 0.025",
        "--placement-mode",
        "surface",
        "--place-clearance",
        "0.003",
        "--transport-z",
        "1.12",
    ]


def parse_result(stdout: str) -> Dict[str, object]:
    for line in stdout.splitlines():
        if line.startswith("result_json:"):
            return json.loads(line.split(":", 1)[1].strip())
    return {}


def run_scene(
    version: str,
    scene_id: str,
    args: argparse.Namespace,
    camera_config: Mapping[str, object],
    seed: int,
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    if config is not None:
        scene = config_util.scenes_by_id(config)[scene_id]
        path = build_config_bddl(
            config,
            scene,
            GENERATED_BDDL_ROOT / "{}_videos".format(config_util.output_name(config)),
        )
        scene_type = str(scene["scene_type"])
        benign_objects = {
            "benign_1": str(scene["benign"][0]),
            "benign_2": str(scene["benign"][1]),
        }
        object_instance, object_category = selected_object(
            scene_type,
            scene_id,
            benign_objects,
            config.get("pick_place", {}),
        )
        camera = config_util.camera_settings(config, scene_type)
        width, height = config_dimensions(config, args)
        video = config_video_values(config, args)
    else:
        path = bddl_path(version, scene_id)
        scene_type, benign_objects = parse_bddl(path)
        object_instance, object_category = selected_object(scene_type, scene_id, benign_objects)
        camera = scene_camera(camera_config, scene_type)
        width = int(args.width or 512)
        height = int(args.height or 512)
        video = {
            "fps": int(args.fps or 20),
            "record_every": int(args.record_every or 2),
            "codec": str(args.video_codec or "libx264"),
            "profile": str(args.video_profile if args.video_profile is not None else "baseline"),
            "pix_fmt": str(args.video_pix_fmt or "yuv420p"),
            "level": str(args.video_level if args.video_level is not None else "3.1"),
            "faststart": True if args.video_faststart is None else bool(args.video_faststart),
        }
    video_path = args.output_root / version / "{}.mp4".format(scene_id)

    base_row: Dict[str, object] = {
        "version": version,
        "scene_id": scene_id,
        "scene_type": scene_type,
        "bddl_path": str(path),
        "selected_object": object_instance,
        "selected_object_category": object_category,
        "video_path": str(video_path),
        "camera": camera,
        "video_settings": video,
    }

    if video_path.exists() and video_path.stat().st_size > 0 and not args.force:
        return {**base_row, "status": "skipped_existing"}

    command = [
        sys.executable,
        str(RUN_PICK_PLACE),
        object_instance,
        "--bddl",
        str(path),
        "--output",
        str(video_path),
        "--camera",
        str(camera["camera"]),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(video["fps"]),
        "--seed",
        str(seed),
        "--record-every",
        str(video["record_every"]),
        "--video-codec",
        str(video["codec"]),
        "--video-profile",
        str(video["profile"]),
        "--video-pix-fmt",
        str(video["pix_fmt"]),
        "--video-level",
        str(video["level"]),
        "--camera-distance-scale",
        str(camera["distance_scale"]),
        "--camera-offset-x",
        str(camera["offset_x"]),
        "--camera-offset-y",
        str(camera["offset_y"]),
        "--camera-offset-z",
        str(camera["offset_z"]),
    ]
    if not bool(video["faststart"]):
        command.append("--no-video-faststart")
    command.extend(target_args(scene_type))
    result = subprocess.run(
        command,
        cwd=str(LIBERO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    row = {
        **base_row,
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if result.returncode == 0 and video_path.exists() and video_path.stat().st_size > 0:
        row.update(parse_result(result.stdout))
        row["status"] = "success"
        row["video_size_bytes"] = video_path.stat().st_size
    else:
        row["status"] = "failed"
    return row


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def replace_rows(
    existing_rows: Sequence[Dict[str, object]],
    new_rows: Sequence[Dict[str, object]],
    key_fields: Sequence[str],
) -> List[Dict[str, object]]:
    new_by_key = {tuple(row.get(field) for field in key_fields): row for row in new_rows}
    seen = set()
    rows: List[Dict[str, object]] = []
    for row in existing_rows:
        key = tuple(row.get(field) for field in key_fields)
        if key in new_by_key:
            rows.append(new_by_key[key])
            seen.add(key)
        else:
            rows.append(row)
    for row in new_rows:
        key = tuple(row.get(field) for field in key_fields)
        if key not in seen:
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.config is not None and len(args.versions) != 1:
        raise SystemExit("--config can only be used with exactly one version")
    camera_config = read_camera_config(args.camera_config.expanduser().resolve())
    selected_scene_ids = scene_ids(args)
    partial_update = bool(args.scene_ids)
    all_rows: List[Dict[str, object]] = []

    for version in args.versions:
        config = config_util.load_config(
            version=version,
            mode=args.config_mode,
            config=args.config,
            config_dir=args.config_dir,
        )
        seed_value = int(args.seed) if args.seed is not None else int(config.get("seed", 7))
        seeds = config_util.scene_seeds(int(config.get("count", 50)), seed_value)
        version_rows: List[Dict[str, object]] = []
        for scene_id in selected_scene_ids:
            row = run_scene(version, scene_id, args, camera_config, seeds[scene_id], config)
            version_rows.append(row)
            all_rows.append(row)
            print("{} {} {} {}".format(version, scene_id, row["selected_object"], row["status"]), flush=True)
            if row["status"] == "failed" and args.stop_on_error:
                write_jsonl(args.output_root / version / "metadata.jsonl", version_rows)
                write_jsonl(args.output_root / "metadata.jsonl", all_rows)
                raise SystemExit("failed: {} {}".format(version, scene_id))
        if partial_update:
            existing_rows = read_jsonl(args.output_root / version / "metadata.jsonl")
            version_rows = replace_rows(existing_rows, version_rows, ("scene_id",))
        write_jsonl(args.output_root / version / "metadata.jsonl", version_rows)

    if partial_update:
        existing_rows = read_jsonl(args.output_root / "metadata.jsonl")
        all_rows = replace_rows(existing_rows, all_rows, ("version", "scene_id"))
    write_jsonl(args.output_root / "metadata.jsonl", all_rows)
    success_count = sum(row.get("status") in {"success", "skipped_existing"} for row in all_rows)
    failed_count = sum(row.get("status") == "failed" for row in all_rows)
    print("rows:", len(all_rows))
    print("success_or_existing:", success_count)
    print("failed:", failed_count)
    print("output_root:", args.output_root)
    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
