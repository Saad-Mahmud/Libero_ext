#!/usr/bin/env python3
"""Recreate the published LIBERO safety dataset configs.

The source of truth for v1-v3 is kept in ``published_dataset_specs/``. For
v4/v5, the source of truth is the master/current JSON config files. This script
renders BDDL scenes, writes local output metadata, and prepares Hugging Face
ImageFolder upload folders for the configured versions.
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import dataset_config as config_util
import generate_scene_dataset as scene_generator


EXTENSION_ROOT = Path(__file__).resolve().parent
LIBERO_ROOT = EXTENSION_ROOT.parents[1]
SPEC_ROOT = EXTENSION_ROOT / "published_dataset_specs"
GENERATED_BDDL_ROOT = EXTENSION_ROOT / "generated_bddl"
OUTPUT_ROOT = EXTENSION_ROOT / "outputs"
DATASET_ROOT = EXTENSION_ROOT / "datasets"
COMBINED_UPLOAD_DIR = DATASET_ROOT / "libero_safety_combined_upload"
VIDEO_ROOT = EXTENSION_ROOT / "videos" / "libero_safety_pick_place"

PUBLISHED_VERSIONS = ("v1", "v2", "v3", "v4", "v5")
OUTPUT_NAMES = {
    "v1": "libero_safety_v1",
    "v2": "libero_safety_v2",
    "v3": "libero_safety_v3",
    "v4": "libero_safety_v4",
    "v5": "libero_safety_v5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and prepare the published LIBERO safety datasets."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=PUBLISHED_VERSIONS,
        default=list(PUBLISHED_VERSIONS),
        help="Dataset configs to build.",
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--camera-distance-scale", type=float, default=1.15)
    parser.add_argument("--camera-offset-x", type=float, default=0.0)
    parser.add_argument("--camera-offset-y", type=float, default=0.0)
    parser.add_argument("--camera-offset-z", type=float, default=0.0)
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=None,
        help="Optional visualizer-saved JSON with per-scene-type camera configs.",
    )
    parser.add_argument(
        "--config-mode",
        choices=config_util.CONFIG_MODES,
        default="current",
        help="For v4/v5, choose editable current config or canonical master config.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=config_util.CONFIG_ROOT,
        help="Directory containing master/current v4/v5 JSON configs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit config JSON path. Only valid when building one v4/v5 version.",
    )
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--push-hf",
        action="store_true",
        help="Upload the prepared combined folder to Hugging Face.",
    )
    parser.add_argument(
        "--repo-id",
        default="saaduddinM/libero_safety_v1",
        help="Hugging Face dataset repo used with --push-hf.",
    )
    return parser.parse_args()


def normalize_scene_type(value: object) -> str:
    scene_type = str(value).strip()
    if scene_type == "giftbox":
        return "gift_box"
    return scene_type


def load_camera_config(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError("Missing camera config: {}".format(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return data


def camera_settings(args: argparse.Namespace, camera_config: Mapping[str, object], scene_type: str) -> Dict[str, object]:
    base: Dict[str, object] = {
        "camera": args.camera,
        "distance_scale": args.camera_distance_scale,
        "offset_x": args.camera_offset_x,
        "offset_y": args.camera_offset_y,
        "offset_z": args.camera_offset_z,
    }
    scene_types = camera_config.get("scene_types", {}) if isinstance(camera_config, dict) else {}
    saved: object = {}
    if isinstance(scene_types, dict):
        saved = scene_types.get(scene_type) or scene_types.get(scene_type.replace("_", "")) or {}
    if isinstance(saved, dict):
        base.update(saved)
    return base


def scene_seeds(count: int, seed: int) -> Dict[str, int]:
    rng = random.Random(seed)
    return {
        "scene{:03d}".format(index): rng.randrange(0, 2**31)
        for index in range(1, count + 1)
    }


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_spec_metadata(version: str) -> List[Dict[str, object]]:
    path = SPEC_ROOT / version / "metadata.jsonl"
    if not path.exists():
        raise FileNotFoundError("Missing metadata spec: {}".format(path))
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 50:
        raise ValueError("{} expected 50 metadata rows, got {}".format(version, len(rows)))
    return rows


def simple_output_record(row: Mapping[str, object]) -> Dict[str, object]:
    record: Dict[str, object] = {
        "scene": row["scene_name"],
        "benign": row["safe_objects"],
        "hazard": row["unsafe_object"],
    }
    for key in ("scene_id", "positions"):
        if key in row:
            record[key] = row[key]
    for key, value in row.items():
        if str(key).startswith("reference_object"):
            record[str(key)] = value
    return record


def write_output_metadata(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    simple_rows = [simple_output_record(row) for row in rows]
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in simple_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene", "benign", "hazard"])
        writer.writeheader()
        for row in simple_rows:
            writer.writerow(
                {
                    "scene": row["scene"],
                    "benign": "; ".join(str(item) for item in row["benign"]),
                    "hazard": row["hazard"],
                }
            )


def copy_bddls(version: str, bddl_dir: Path) -> None:
    source_dir = SPEC_ROOT / version / "bddl"
    if not source_dir.exists():
        raise FileNotFoundError("Missing BDDL spec directory: {}".format(source_dir))
    clean_dir(bddl_dir)
    for index in range(1, 51):
        scene_id = "scene{:03d}".format(index)
        source_path = source_dir / "{}.bddl".format(scene_id)
        if not source_path.exists():
            raise FileNotFoundError("Missing BDDL spec: {}".format(source_path))
        shutil.copy2(source_path, bddl_dir / source_path.name)


def render_scene(
    bddl_path: Path,
    image_path: Path,
    args: argparse.Namespace,
    seed: int,
    camera: Mapping[str, object],
) -> None:
    command = [
        sys.executable,
        str(EXTENSION_ROOT / "run.py"),
        str(bddl_path),
        "--mode",
        "image",
        "--camera",
        str(camera["camera"]),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--camera-distance-scale",
        str(camera["distance_scale"]),
        "--camera-offset-x",
        str(camera["offset_x"]),
        "--camera-offset-y",
        str(camera["offset_y"]),
        "--camera-offset-z",
        str(camera["offset_z"]),
        "--output",
        str(image_path),
        "--seed",
        str(seed),
    ]
    result = subprocess.run(
        command,
        cwd=str(LIBERO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "render failed for {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                bddl_path,
                result.stdout,
                result.stderr,
            )
        )


def render_version(
    version: str,
    bddl_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    seeds: Mapping[str, int],
    rows: Sequence[Mapping[str, object]],
    camera_config: Mapping[str, object],
) -> None:
    clean_dir(output_dir)
    rows_by_scene = {str(row["scene_id"]): row for row in rows}
    for index in range(1, args.count + 1):
        scene_id = "scene{:03d}".format(index)
        row = rows_by_scene[scene_id]
        scene_type = normalize_scene_type(row["scene_name"])
        render_scene(
            bddl_dir / "{}.bddl".format(scene_id),
            output_dir / "{}.png".format(scene_id),
            args,
            seeds[scene_id],
            camera_settings(args, camera_config, scene_type),
        )
        print("rendered:", version, scene_id)


def render_args_from_config(config: Mapping[str, object], camera: Mapping[str, object]) -> argparse.Namespace:
    image = config_util.image_settings(config)
    return argparse.Namespace(
        camera=str(camera["camera"]),
        width=int(image["width"]),
        height=int(image["height"]),
        camera_distance_scale=float(camera["distance_scale"]),
        camera_offset_x=float(camera["offset_x"]),
        camera_offset_y=float(camera["offset_y"]),
        camera_offset_z=float(camera["offset_z"]),
    )


def write_config_output_metadata(output_dir: Path, config: Mapping[str, object]) -> None:
    rows = []
    for row in config_util.config_metadata_rows(config):
        rows.append(
            {
                "scene": row["scene_name"],
                "benign": row["safe_objects"],
                "hazard": row["unsafe_object"],
                **row,
            }
        )
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene", "benign", "hazard"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scene": row["scene"],
                    "benign": "; ".join(str(item) for item in row["benign"]),
                    "hazard": row["hazard"],
                }
            )


def build_config_bddls(config: Mapping[str, object], bddl_dir: Path) -> None:
    clean_dir(bddl_dir)
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


def render_config_version(
    config: Mapping[str, object],
    bddl_dir: Path,
    output_dir: Path,
) -> None:
    clean_dir(output_dir)
    version = str(config["version"])
    for scene in config.get("scenes", []):
        scene_id = str(scene["scene_id"])
        camera = config_util.camera_settings(config, str(scene["scene_type"]))
        render_scene(
            bddl_dir / "{}.bddl".format(scene_id),
            output_dir / "{}.png".format(scene_id),
            render_args_from_config(config, camera),
            config_util.scene_seed(config, scene_id),
            camera,
        )
        print("rendered:", version, scene_id)


def load_version_config(version: str, args: argparse.Namespace) -> Dict[str, object]:
    if args.config is not None and len(args.versions) != 1:
        raise ValueError("--config can only be used with exactly one version")
    return config_util.load_config(
        version=version,
        mode=args.config_mode,
        config=args.config,
        config_dir=args.config_dir,
    )


def build_config_version(version: str, args: argparse.Namespace) -> None:
    config = load_version_config(version, args)
    output_name = config_util.output_name(config)
    bddl_dir = GENERATED_BDDL_ROOT / output_name
    output_dir = OUTPUT_ROOT / output_name

    build_config_bddls(config, bddl_dir)
    if args.skip_render:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        render_config_version(config, bddl_dir, output_dir)
    write_config_output_metadata(output_dir, config)
    print("metadata:", version, output_dir / "metadata.jsonl")


def build_version(version: str, args: argparse.Namespace, seeds: Mapping[str, int]) -> None:
    if version in config_util.CONFIG_VERSIONS:
        build_config_version(version, args)
        return

    output_name = OUTPUT_NAMES[version]
    bddl_dir = GENERATED_BDDL_ROOT / output_name
    output_dir = OUTPUT_ROOT / output_name
    rows = read_spec_metadata(version)

    copy_bddls(version, bddl_dir)
    if args.skip_render:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        render_version(version, bddl_dir, output_dir, args, seeds, rows, load_camera_config(args.camera_config))
    write_output_metadata(output_dir, rows)
    print("metadata:", version, output_dir / "metadata.jsonl")


def dataset_card(versions: Sequence[str]) -> str:
    config_blocks = []
    for version in versions:
        config_blocks.append(
            "- config_name: {version}\n"
            "  data_files:\n"
            "  - split: train\n"
            "    path: {version}/data/train/**".format(version=version)
        )
    config_text = "\n".join(config_blocks)
    return """---
configs:
{config_text}
license: other
task_categories:
- visual-question-answering
- image-classification
language:
- en
size_categories:
- n<1K
pretty_name: LIBERO Safety
tags:
- libero
- robotics
- safety
- mujoco
- synthetic
---

# LIBERO Safety

This repository contains 50-scene synthetic LIBERO safety validation sets.

- `v1`: original LIBERO safety image set.
- `v2`: regenerated set using the latest scene configuration and saved manual positions.
- `v3`: same scenes and labels as `v2`, with a static non-colliding MuJoCo white cutting board fixture added on the table; original V2 object and fixture states are unchanged.
- `v4`: corrected zoomed non-reference set with pickable target objects for video rollouts.
- `v5`: corrected zoomed white-cutting-board reference set with pickable target objects for video rollouts.

Each config has a `train` split with rendered scene images and metadata.

Columns:

- `image`: rendered LIBERO scene image, inferred from `file_name` by Hugging Face ImageFolder.
- `scene_id`: deterministic scene identifier from `scene001` onward.
- `scene_name`: scene context name.
- `safe_objects`: objects that naturally belong in the scene context.
- `unsafe_object`: one object that does not belong in the scene context.
- `video`: MP4 pick-place rollout for the same scene. This is present in `v4` and `v5`.
- `reference_object`: fixed visual reference object. This is present in `v3` and `v5`.
- `reference_object_category`: reference object category. This is present in `v3` and `v5`.
- `reference_object_position`: table position for the reference object. This is present in `v3` and `v5`.

Scene counts for each config:

- `giftbox`: 17
- `microwave`: 16
- `stove`: 17

Load with:

```python
from datasets import load_dataset

v1 = load_dataset("saaduddinM/libero_safety_v1", "v1", split="train")
v2 = load_dataset("saaduddinM/libero_safety_v1", "v2", split="train")
v3 = load_dataset("saaduddinM/libero_safety_v1", "v3", split="train")
v4 = load_dataset("saaduddinM/libero_safety_v1", "v4", split="train")
v5 = load_dataset("saaduddinM/libero_safety_v1", "v5", split="train")
```
""".format(config_text=config_text)


def prepare_version(version: str) -> None:
    output_name = OUTPUT_NAMES[version]
    output_dir = OUTPUT_ROOT / output_name
    upload_dir = DATASET_ROOT / "libero_safety_{}_upload".format(version)
    command = [
        sys.executable,
        str(EXTENSION_ROOT / "prepare_hf_scene_dataset.py"),
        "--input-dir",
        str(output_dir),
        "--output-dir",
        str(upload_dir),
        "--pretty-name",
        "LIBERO Safety {}".format(version.upper()),
    ]
    video_dir = VIDEO_ROOT / version
    if version in {"v4", "v5"} and video_dir.exists():
        command.extend(["--video-dir", str(video_dir)])
    result = subprocess.run(
        command,
        cwd=str(LIBERO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "prepare failed for {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                version,
                result.stdout,
                result.stderr,
            )
        )
    print(result.stdout.strip())

    target = COMBINED_UPLOAD_DIR / version
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(upload_dir, target)


def prepare_combined(versions: Sequence[str]) -> None:
    COMBINED_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for version in versions:
        prepare_version(version)
    card_versions = [
        version for version in PUBLISHED_VERSIONS if (COMBINED_UPLOAD_DIR / version).exists()
    ]
    (COMBINED_UPLOAD_DIR / "README.md").write_text(
        dataset_card(card_versions or versions),
        encoding="utf-8",
    )
    print("combined_upload_dir:", COMBINED_UPLOAD_DIR)


def push_hf(repo_id: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(COMBINED_UPLOAD_DIR),
        commit_message="Publish LIBERO safety dataset configs",
    )
    return info.commit_url


def main() -> None:
    args = parse_args()
    versions = list(dict.fromkeys(args.versions))
    seeds = scene_seeds(args.count, args.seed)
    for version in versions:
        build_version(version, args, seeds)
    if not args.skip_prepare:
        prepare_combined(versions)
    if args.push_hf:
        print("pushed:", push_hf(args.repo_id))


if __name__ == "__main__":
    main()
