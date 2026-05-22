#!/usr/bin/env python3
"""Recreate the published LIBERO safety dataset configs.

The source of truth for the published sets is kept in
``published_dataset_specs/``. Each version has canonical BDDL files and
metadata. This script renders those BDDLs, writes local output metadata, and
prepares Hugging Face ImageFolder upload folders for configs v1, v2, and v3.
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


EXTENSION_ROOT = Path(__file__).resolve().parent
LIBERO_ROOT = EXTENSION_ROOT.parents[1]
SPEC_ROOT = EXTENSION_ROOT / "published_dataset_specs"
GENERATED_BDDL_ROOT = EXTENSION_ROOT / "generated_bddl"
OUTPUT_ROOT = EXTENSION_ROOT / "outputs"
DATASET_ROOT = EXTENSION_ROOT / "datasets"
COMBINED_UPLOAD_DIR = DATASET_ROOT / "libero_safety_combined_upload"

PUBLISHED_VERSIONS = ("v1", "v2", "v3")
OUTPUT_NAMES = {
    "v1": "libero_safety_v1",
    "v2": "libero_safety_v2",
    "v3": "libero_safety_v3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and prepare the published LIBERO safety v1/v2/v3 datasets."
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
) -> None:
    command = [
        sys.executable,
        str(EXTENSION_ROOT / "run.py"),
        str(bddl_path),
        "--mode",
        "image",
        "--camera",
        args.camera,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--camera-distance-scale",
        str(args.camera_distance_scale),
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
) -> None:
    clean_dir(output_dir)
    for index in range(1, args.count + 1):
        scene_id = "scene{:03d}".format(index)
        render_scene(
            bddl_dir / "{}.bddl".format(scene_id),
            output_dir / "{}.png".format(scene_id),
            args,
            seeds[scene_id],
        )
        print("rendered:", version, scene_id)


def build_version(version: str, args: argparse.Namespace, seeds: Mapping[str, int]) -> None:
    output_name = OUTPUT_NAMES[version]
    bddl_dir = GENERATED_BDDL_ROOT / output_name
    output_dir = OUTPUT_ROOT / output_name
    rows = read_spec_metadata(version)

    copy_bddls(version, bddl_dir)
    if args.skip_render:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        render_version(version, bddl_dir, output_dir, args, seeds)
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

This repository contains three 50-scene synthetic LIBERO safety validation sets.

- `v1`: original LIBERO safety image set.
- `v2`: regenerated set using the latest scene configuration and saved manual positions.
- `v3`: same scenes and labels as `v2`, with a static non-colliding MuJoCo white cutting board fixture added on the table; original V2 object and fixture states are unchanged.

Each config has a `train` split with rendered scene images and metadata.

Columns:

- `image`: rendered LIBERO scene image, inferred from `file_name` by Hugging Face ImageFolder.
- `scene_id`: deterministic scene identifier from `scene001` onward.
- `scene_name`: scene context name.
- `safe_objects`: objects that naturally belong in the scene context.
- `unsafe_object`: one object that does not belong in the scene context.
- `reference_object`: fixed visual reference object. This is present in `v3`.
- `reference_object_category`: reference object category. This is present in `v3`.
- `reference_object_position`: table position for the reference object. This is present in `v3`.

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
    for stale in ("v1", "v2", "v3", "v4", "v5"):
        if stale not in versions:
            path = COMBINED_UPLOAD_DIR / stale
            if path.exists():
                shutil.rmtree(path)
    for version in versions:
        prepare_version(version)
    (COMBINED_UPLOAD_DIR / "README.md").write_text(
        dataset_card(versions),
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
        commit_message="Publish LIBERO safety v1 v2 v3",
        delete_patterns=["v4/**", "v5/**"],
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
