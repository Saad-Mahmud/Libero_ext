#!/usr/bin/env python3
"""Config helpers for LIBERO safety dataset v4/v5 generation."""

from __future__ import annotations

import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


EXTENSION_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = EXTENSION_ROOT / "configs"
CONFIG_MODES = ("master", "current")
CONFIG_VERSIONS = ("v4", "v5")


def normalize_version(value: str) -> str:
    version = str(value).strip().lower()
    if version not in CONFIG_VERSIONS:
        raise ValueError("expected one of {}, got {}".format(", ".join(CONFIG_VERSIONS), value))
    return version


def normalize_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in CONFIG_MODES:
        raise ValueError("expected config mode master or current, got {}".format(value))
    return mode


def infer_version(*values: object) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        match = re.search(r"(v[45])", str(value).lower())
        if match:
            return match.group(1)
    return None


def config_path(version: str, mode: str = "current", config_dir: Path = CONFIG_ROOT) -> Path:
    return Path(config_dir).expanduser().resolve() / "{}_config_{}.json".format(
        normalize_mode(mode),
        normalize_version(version),
    )


def resolve_config_path(
    version: Optional[str] = None,
    mode: str = "current",
    config: Optional[Path] = None,
    config_dir: Path = CONFIG_ROOT,
) -> Path:
    if config is not None:
        return Path(config).expanduser().resolve()
    if version is None:
        raise ValueError("version is required when --config is not provided")
    return config_path(version, mode, config_dir)


def load_config(
    version: Optional[str] = None,
    mode: str = "current",
    config: Optional[Path] = None,
    config_dir: Path = CONFIG_ROOT,
) -> Dict[str, Any]:
    path = resolve_config_path(version, mode, config, config_dir)
    if not path.exists():
        raise FileNotFoundError("Missing dataset config: {}".format(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    data["_config_path"] = str(path)
    validate_config(data)
    return data


def write_config(config: Mapping[str, Any], path: Optional[Path] = None) -> None:
    raw_path = path or config.get("_config_path")
    if not raw_path:
        raise ValueError("config path is required")
    output_path = Path(raw_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {key: value for key, value in dict(config).items() if key != "_config_path"}
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_config(config: Mapping[str, Any]) -> None:
    version = normalize_version(str(config.get("version", "")))
    scenes = config.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != int(config.get("count", 50)):
        raise ValueError("{} config must contain count matching scenes".format(version))
    seen = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ValueError("{} config scenes must be objects".format(version))
        scene_id = str(scene.get("scene_id", ""))
        if not re.match(r"^scene\d{3}$", scene_id):
            raise ValueError("{} has invalid scene_id {}".format(version, scene_id))
        if scene_id in seen:
            raise ValueError("{} has duplicate {}".format(version, scene_id))
        seen.add(scene_id)
        if scene.get("scene_type") not in {"gift_box", "stove", "microwave"}:
            raise ValueError("{} has invalid scene_type {}".format(scene_id, scene.get("scene_type")))
        if len(scene.get("benign", [])) != 2:
            raise ValueError("{} must have exactly two benign objects".format(scene_id))
        if not scene.get("dangerous"):
            raise ValueError("{} missing dangerous object".format(scene_id))


def scenes_by_id(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(scene["scene_id"]): deepcopy(scene) for scene in config.get("scenes", [])}


def scene_plan_from_config(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    plan: Dict[str, Dict[str, Any]] = {}
    for scene in config.get("scenes", []):
        entry = {
            "scene_id": scene["scene_id"],
            "scene_type": scene["scene_type"],
            "benign": list(scene["benign"]),
            "dangerous": scene["dangerous"],
            "positions": deepcopy(scene.get("positions", {})),
        }
        plan[str(scene["scene_id"])] = entry
    return plan


def scene_number(scene_id: str) -> int:
    value = str(scene_id).strip().lower()
    if value.startswith("scene"):
        value = value[5:]
    return int(value)


def scene_seeds(count: int, seed: int) -> Dict[str, int]:
    rng = random.Random(seed)
    return {
        "scene{:03d}".format(index): rng.randrange(0, 2**31)
        for index in range(1, count + 1)
    }


def scene_seed(config: Mapping[str, Any], scene_id: str) -> int:
    return scene_seeds(int(config.get("count", 50)), int(config.get("seed", 7)))[scene_id]


def image_settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    image = config.get("image", {})
    if not isinstance(image, dict):
        image = {}
    return {
        "camera": str(image.get("camera", "frontview")),
        "width": int(image.get("width", 512)),
        "height": int(image.get("height", 512)),
    }


def camera_settings(config: Mapping[str, Any], scene_type: str) -> Dict[str, Any]:
    camera = config.get("camera", {})
    if not isinstance(camera, dict):
        camera = {}
    defaults = camera.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    scene_types = camera.get("scene_types", {})
    if not isinstance(scene_types, dict):
        scene_types = {}
    saved = scene_types.get(scene_type) or scene_types.get(str(scene_type).replace("_", "")) or {}
    if not isinstance(saved, dict):
        saved = {}
    merged = {
        "camera": defaults.get("camera", image_settings(config)["camera"]),
        "distance_scale": defaults.get("distance_scale", 1.15),
        "offset_x": defaults.get("offset_x", 0.0),
        "offset_y": defaults.get("offset_y", 0.0),
        "offset_z": defaults.get("offset_z", 0.0),
    }
    merged.update(saved)
    return {
        "camera": str(merged["camera"]),
        "distance_scale": float(merged["distance_scale"]),
        "offset_x": float(merged["offset_x"]),
        "offset_y": float(merged["offset_y"]),
        "offset_z": float(merged["offset_z"]),
    }


def video_settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    video = config.get("video", {})
    if not isinstance(video, dict):
        video = {}
    return {
        "fps": int(video.get("fps", 20)),
        "record_every": int(video.get("record_every", 2)),
        "codec": str(video.get("codec", "libx264")),
        "profile": str(video.get("profile", "baseline")),
        "pix_fmt": str(video.get("pix_fmt", "yuv420p")),
        "level": str(video.get("level", "3.1")),
        "faststart": bool(video.get("faststart", True)),
    }


def output_name(config: Mapping[str, Any]) -> str:
    return str(config.get("output_name") or "libero_safety_{}".format(config["version"]))


def dataset_scene_name(scene_type: str) -> str:
    return "giftbox" if scene_type == "gift_box" else scene_type


def reference_from_scene_dict(scene: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    reference = scene.get("reference_object")
    return deepcopy(reference) if isinstance(reference, dict) else None


def reference_object_from_scene(
    generator: Any,
    scene: Mapping[str, Any],
    positions: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Optional[Any]:
    reference = reference_from_scene_dict(scene)
    if reference is None:
        return None
    position = deepcopy(reference.get("position", {}))
    instance_name = str(reference.get("instance", "reference_object_1"))
    if positions and instance_name in positions:
        override = positions[instance_name]
        position["x"] = float(override["x"])
        position["y"] = float(override["y"])
        if "yaw" in override:
            position["yaw"] = float(override["yaw"])
    kind = str(position.get("kind", reference.get("kind", "fixture")))
    if kind == "static_fixture":
        kind = "fixture"
    return generator.ReferenceObject(
        instance_name=instance_name,
        category_name=str(reference.get("bddl_category") or reference.get("category")),
        display_name=str(reference.get("name") or reference.get("category")),
        center=(float(position["x"]), float(position["y"])),
        yaw=float(position.get("yaw", 0.0)),
        radius=float(position.get("radius", reference.get("radius", 0.075))),
        kind=kind,
        role=str(reference.get("role", "fixed simulated LIBERO reference fixture")),
    )


def public_reference_fields(scene: Mapping[str, Any]) -> Dict[str, Any]:
    reference = reference_from_scene_dict(scene)
    if reference is None:
        return {}
    position = deepcopy(reference.get("position", {}))
    return {
        "reference_object": reference.get("name"),
        "reference_object_category": reference.get("category"),
        "reference_object_bddl_category": reference.get("bddl_category"),
        "reference_object_instance": reference.get("instance"),
        "reference_object_position": position,
        "reference_object_role": reference.get("role"),
    }


def config_metadata_rows(config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for scene in config.get("scenes", []):
        row = {
            "file_name": "{}.png".format(scene["scene_id"]),
            "scene_id": scene["scene_id"],
            "scene_name": scene.get("scene_name") or dataset_scene_name(str(scene["scene_type"])),
            "safe_objects": list(scene.get("safe_objects", [])),
            "unsafe_object": scene.get("unsafe_object"),
            "positions": deepcopy(scene.get("positions", {})),
        }
        row.update(public_reference_fields(scene))
        rows.append(row)
    return rows


def update_scene_positions(
    config: Dict[str, Any],
    scene_id: str,
    positions: Mapping[str, Mapping[str, float]],
) -> None:
    for scene in config.get("scenes", []):
        if scene.get("scene_id") != scene_id:
            continue
        saved_positions = deepcopy(dict(positions))
        reference = scene.get("reference_object")
        if isinstance(reference, dict):
            instance = str(reference.get("instance", "reference_object_1"))
            if instance in saved_positions:
                reference_position = dict(reference.get("position", {}))
                moved = saved_positions.pop(instance)
                reference_position["x"] = float(moved["x"])
                reference_position["y"] = float(moved["y"])
                if "yaw" in moved:
                    reference_position["yaw"] = float(moved["yaw"])
                reference["position"] = reference_position
        scene["positions"] = saved_positions
        return
    raise ValueError("{} not found in config".format(scene_id))


def update_scene_type_camera(config: Dict[str, Any], scene_type: str, camera: Mapping[str, Any]) -> None:
    config.setdefault("camera", {}).setdefault("scene_types", {})[scene_type] = {
        "camera": str(camera["camera"]),
        "distance_scale": float(camera["distance_scale"]),
        "offset_x": float(camera["offset_x"]),
        "offset_y": float(camera["offset_y"]),
        "offset_z": float(camera["offset_z"]),
    }
