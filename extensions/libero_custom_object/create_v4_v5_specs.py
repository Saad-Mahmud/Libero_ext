#!/usr/bin/env python3
"""Create v4/v5 published specs from v2/v3 with corrected pickable objects."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, Mapping

import dataset_config as config_util
import generate_scene_dataset as scene_generator


EXTENSION_ROOT = Path(__file__).resolve().parent
SPEC_ROOT = EXTENSION_ROOT / "published_dataset_specs"
SCENE_PLAN_PATH = EXTENSION_ROOT / "scene_object_plan.json"

VERSION_SOURCES = {
    "v4": "v2",
    "v5": "v3",
}

CATEGORY_DISPLAY = {
    "custom_baseball_ball": "baseball",
    "custom_toy_ball": "toy ball",
    "custom_toy_block": "toy block",
    "custom_toy_car": "toy car",
    "custom_toy_train": "toy train",
    "chefmate_8_frypan": "skillet",
    "custom_moka_pot_small": "coffee pot",
    "glazed_rim_porcelain_ramekin": "ramekin",
    "custom_stove_mini_metal_pot": "mini metal pot",
    "custom_stove_mini_saucepan": "mini saucepan",
    "custom_stove_mini_kettle": "mini kettle",
    "custom_rc_cup_ov_cup_2": "cup",
    "custom_rc_dumpling_ag_dumpling_0": "dumpling",
    "custom_rc_egg_ov_egg_0": "egg",
    "custom_rc_tomato_ov_tomato_0": "tomato",
}

SCENE_REPLACEMENTS: Dict[str, Dict[str, str]] = {
    "scene018": {"benign_2": "custom_moka_pot_small"},
    "scene021": {"benign_2": "custom_moka_pot_small"},
    "scene026": {"benign_2": "custom_moka_pot_small"},
    "scene028": {"benign_2": "custom_moka_pot_small"},
    "scene032": {"benign_2": "custom_moka_pot_small"},
    "scene001": {"benign_1": "custom_toy_train", "benign_2": "custom_toy_ball"},
    "scene006": {"benign_1": "custom_toy_ball", "benign_2": "custom_baseball_ball"},
    "scene010": {"benign_1": "custom_toy_block", "benign_2": "custom_toy_car"},
    "scene014": {"benign_1": "custom_toy_block", "benign_2": "custom_baseball_ball"},
    "scene037": {"benign_2": "custom_rc_egg_ov_egg_0"},
    "scene038": {"benign_1": "custom_rc_dumpling_ag_dumpling_0"},
    "scene040": {"benign_2": "custom_rc_dumpling_ag_dumpling_0"},
    "scene041": {"benign_2": "custom_rc_egg_ov_egg_0"},
    "scene043": {"benign_1": "custom_rc_cup_ov_cup_2"},
    "scene046": {"benign_1": "custom_rc_tomato_ov_tomato_0"},
    "scene047": {"benign_1": "custom_rc_cup_ov_cup_2"},
}

STOVE_SCENE_IDS = {"scene{:03d}".format(index) for index in range(18, 35)}
STOVE_DISPLAY_REPLACEMENTS = {
    "frying pan": "skillet",
    "moka pot": "coffee pot",
}

POSITION_REGION_NAMES = {
    "gift_box_1": "gift_box_init_region",
    "gift_box_lid_1": "gift_box_lid_init_region",
    "flat_stove_1": "flat_stove_init_region",
    "microwave_1": "microwave_init_region",
    "plate_1": "plate_init_region",
    "reference_cutting_board_1": "reference_object_init_region",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create canonical v4/v5 specs from v2/v3.")
    parser.add_argument("--versions", nargs="+", choices=sorted(VERSION_SOURCES), default=sorted(VERSION_SOURCES))
    parser.add_argument("--force", action="store_true", help="Replace existing v4/v5 spec directories.")
    parser.add_argument("--config-mode", choices=config_util.CONFIG_MODES, default="current")
    parser.add_argument("--config-dir", type=Path, default=config_util.CONFIG_ROOT)
    parser.add_argument("--config", type=Path, default=None, help="Explicit config JSON path for a single version.")
    return parser.parse_args()


def replace_bddl_objects(path: Path, replacements: Mapping[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    for instance, category in replacements.items():
        pattern = r"(^\s*{}\s+-\s+)[A-Za-z0-9_]+\s*$".format(re.escape(instance))
        text, count = re.subn(pattern, r"\1{}".format(category), text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise ValueError("Could not replace {} in {}".format(instance, path))
    path.write_text(text, encoding="utf-8")


def replace_bddl_category(path: Path, old_category: str, new_category: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = r"(^\s*[A-Za-z0-9_]+\s+-\s+){}\s*$".format(re.escape(old_category))
    text, count = re.subn(pattern, r"\1{}".format(new_category), text, flags=re.MULTILINE)
    if count < 1:
        raise ValueError("Could not replace {} in {}".format(old_category, path))
    path.write_text(text, encoding="utf-8")


def load_scene_positions() -> Dict[str, Dict[str, Dict[str, float]]]:
    if not SCENE_PLAN_PATH.exists():
        return {}
    data = json.loads(SCENE_PLAN_PATH.read_text(encoding="utf-8"))
    positions_by_scene: Dict[str, Dict[str, Dict[str, float]]] = {}
    for entry in data.get("scenes", []):
        scene_id = entry.get("scene_id")
        positions = entry.get("positions")
        if not scene_id or not isinstance(positions, dict):
            continue
        normalized: Dict[str, Dict[str, float]] = {}
        for instance_name, position in positions.items():
            if not isinstance(position, dict) or "x" not in position or "y" not in position:
                continue
            row = {"x": float(position["x"]), "y": float(position["y"])}
            if "yaw" in position:
                row["yaw"] = float(position["yaw"])
            normalized[str(instance_name)] = row
        positions_by_scene[str(scene_id)] = normalized
    return positions_by_scene


def region_name_for_instance(instance_name: str) -> str:
    if instance_name in POSITION_REGION_NAMES:
        return POSITION_REGION_NAMES[instance_name]
    if instance_name in {"benign_1", "benign_2"}:
        return "{}_slot".format(instance_name)
    if instance_name == "dangerous_1":
        return "dangerous_slot"
    return "{}_init_region".format(instance_name)


def replace_region_position(path: Path, region_name: str, position: Mapping[str, float]) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_region = False
    range_pending = False
    yaw_pending = False
    range_replaced = False
    yaw_replaced = "yaw" not in position
    region_pattern = re.compile(r"\(\s*{}\b".format(re.escape(region_name)))

    for index, line in enumerate(lines):
        if not in_region and region_pattern.search(line):
            in_region = True
            continue
        if not in_region:
            continue

        if "(:ranges" in line:
            range_pending = True
            continue
        if range_pending and line.strip().startswith("("):
            indent = line[: len(line) - len(line.lstrip())]
            x = float(position["x"])
            y = float(position["y"])
            lines[index] = "{}({:.4f} {:.4f} {:.4f} {:.4f})".format(
                indent,
                x - 0.001,
                y - 0.001,
                x + 0.001,
                y + 0.001,
            )
            range_pending = False
            range_replaced = True
            if yaw_replaced:
                break
            continue

        if "(:yaw_rotation" in line:
            yaw_pending = True
            continue
        if yaw_pending and line.strip().startswith("("):
            indent = line[: len(line) - len(line.lstrip())]
            yaw = float(position["yaw"])
            lines[index] = "{}({:.6f} {:.6f})".format(indent, yaw, yaw)
            yaw_pending = False
            yaw_replaced = True
            if range_replaced:
                break

    if not range_replaced:
        return False
    if not yaw_replaced:
        raise ValueError("Could not replace yaw for {} in {}".format(region_name, path))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def apply_saved_positions(target: Path) -> None:
    positions_by_scene = load_scene_positions()
    for scene_id, positions in positions_by_scene.items():
        bddl_path = target / "bddl" / "{}.bddl".format(scene_id)
        if not bddl_path.exists():
            continue
        for instance_name, position in positions.items():
            replace_region_position(bddl_path, region_name_for_instance(instance_name), position)


def update_safe_objects(row: Dict[str, object], replacements: Mapping[str, str]) -> None:
    safe_objects = list(row.get("safe_objects", []))
    if len(safe_objects) != 2:
        raise ValueError("{} expected exactly two safe objects".format(row.get("scene_id")))
    for instance, category in replacements.items():
        if instance not in {"benign_1", "benign_2"}:
            continue
        index = 0 if instance == "benign_1" else 1
        safe_objects[index] = CATEGORY_DISPLAY[category]
    row["safe_objects"] = safe_objects


def update_stove_safe_objects(row: Dict[str, object]) -> None:
    row["safe_objects"] = [
        STOVE_DISPLAY_REPLACEMENTS.get(str(item), str(item))
        for item in list(row.get("safe_objects", []))
    ]


def read_jsonl(path: Path) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_version(version: str, force: bool) -> None:
    source = SPEC_ROOT / VERSION_SOURCES[version]
    target = SPEC_ROOT / version
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists():
        if not force:
            raise FileExistsError("{} exists; pass --force".format(target))
        shutil.rmtree(target)
    shutil.copytree(source, target)

    for scene_id, replacements in SCENE_REPLACEMENTS.items():
        replace_bddl_objects(target / "bddl" / "{}.bddl".format(scene_id), replacements)
    for scene_id in STOVE_SCENE_IDS:
        bddl_path = target / "bddl" / "{}.bddl".format(scene_id)
        if re.search(
            r"^\s*[A-Za-z0-9_]+\s+-\s+moka_pot\s*$",
            bddl_path.read_text(encoding="utf-8"),
            re.MULTILINE,
        ):
            replace_bddl_category(bddl_path, "moka_pot", "custom_moka_pot_small")
    apply_saved_positions(target)

    rows = read_jsonl(target / "metadata.jsonl")
    for row in rows:
        scene_id = str(row["scene_id"])
        if scene_id in STOVE_SCENE_IDS:
            update_stove_safe_objects(row)
        replacements = SCENE_REPLACEMENTS.get(scene_id)
        if replacements:
            update_safe_objects(row, replacements)
    write_jsonl(target / "metadata.jsonl", rows)
    validate_version(version)
    print("created:", version, "from", VERSION_SOURCES[version])


def build_config_version(version: str, args: argparse.Namespace) -> None:
    if args.config is not None and len(args.versions) != 1:
        raise ValueError("--config can only be used with exactly one version")
    config = config_util.load_config(
        version=version,
        mode=args.config_mode,
        config=args.config,
        config_dir=args.config_dir,
    )
    target = SPEC_ROOT / version
    if target.exists():
        if not args.force:
            raise FileExistsError("{} exists; pass --force".format(target))
        shutil.rmtree(target)
    bddl_dir = target / "bddl"
    bddl_dir.mkdir(parents=True)
    for scene in config.get("scenes", []):
        scene_id = str(scene["scene_id"])
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
        (bddl_dir / "{}.bddl".format(scene_id)).write_text(bddl, encoding="utf-8")
    write_jsonl(target / "metadata.jsonl", config_util.config_metadata_rows(config))
    validate_version(version)
    print("created:", version, "from", config.get("_config_path"))


def validate_version(version: str) -> None:
    root = SPEC_ROOT / version
    text = "\n".join(path.read_text(encoding="utf-8") for path in (root / "bddl").glob("scene*.bddl"))
    if "custom_toy_drum" in text:
        raise ValueError("{} still contains custom_toy_drum".format(version))
    if "custom_stove_mini_" in text:
        raise ValueError("{} still contains mini stove placeholder objects".format(version))

    rows = read_jsonl(root / "metadata.jsonl")
    if len(rows) != 50:
        raise ValueError("{} expected 50 rows, got {}".format(version, len(rows)))

    for row in rows:
        scene_id = str(row["scene_id"])
        scene_name = str(row["scene_name"])
        safe_objects = row["safe_objects"]
        if scene_name == "microwave":
            if not set(safe_objects).intersection({"egg", "dumpling", "cup", "tomato"}):
                raise ValueError("{} {} lacks an easy-pick microwave object".format(version, scene_id))
        if scene_name == "stove":
            if "moka pot" in safe_objects:
                raise ValueError("{} {} still exposes native moka pot label".format(version, scene_id))
            if not set(safe_objects).intersection({"skillet", "coffee pot", "ramekin"}):
                raise ValueError("{} {} lacks expected stove-safe objects".format(version, scene_id))


def main() -> None:
    args = parse_args()
    for version in args.versions:
        config_path = config_util.resolve_config_path(
            version=version,
            mode=args.config_mode,
            config=args.config,
            config_dir=args.config_dir,
        )
        if config_path.exists():
            build_config_version(version, args)
        else:
            build_version(version, args.force)


if __name__ == "__main__":
    main()
