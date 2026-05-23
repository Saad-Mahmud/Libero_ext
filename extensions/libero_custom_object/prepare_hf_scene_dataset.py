#!/usr/bin/env python3
"""Prepare and optionally upload the scene dataset in HF ImageFolder format."""

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


EXTENSION_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = EXTENSION_ROOT / "outputs" / "scene_dataset_v1"
DEFAULT_OUTPUT_DIR = EXTENSION_ROOT / "datasets" / "libero_safety_v1_upload"
DEFAULT_SCENE_PLAN = EXTENSION_ROOT / "scene_object_plan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Hugging Face ImageFolder upload folder for scene_dataset_v1."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=None, help="Optional HF dataset repo id, e.g. user/libero_safety_v1.")
    parser.add_argument("--scene-plan", type=Path, default=DEFAULT_SCENE_PLAN)
    parser.add_argument("--video-dir", type=Path, default=None, help="Optional folder with scene###.mp4 videos to include.")
    parser.add_argument("--video-column", default="video", help="Metadata column name for included videos.")
    parser.add_argument("--pretty-name", default=None)
    parser.add_argument("--push", action="store_true", help="Upload the prepared folder to Hugging Face.")
    parser.add_argument(
        "--replace-repo-files",
        action="store_true",
        help="When pushing, delete existing remote files before uploading this dataset-only folder.",
    )
    parser.add_argument("--private", action="store_true", help="Create the HF dataset repo as private.")
    return parser.parse_args()


def infer_pretty_name(output_dir: Path, explicit_name: Optional[str]) -> str:
    if explicit_name:
        return explicit_name
    stem = output_dir.name
    if stem.endswith("_upload"):
        stem = stem[: -len("_upload")]
    lowered = stem.lower()
    for version in ("v5", "v4", "v3", "v2", "v1"):
        if version in lowered:
            return "LIBERO Safety {}".format(version.upper())
    return "LIBERO Safety"


def read_scene_positions(scene_plan: Path) -> Dict[str, Dict[str, object]]:
    if not scene_plan.exists():
        return {}
    data = json.loads(scene_plan.read_text(encoding="utf-8"))
    positions: Dict[str, Dict[str, object]] = {}
    for entry in data.get("scenes", []):
        scene_id = entry.get("scene_id")
        scene_positions = entry.get("positions")
        if scene_id and scene_positions:
            positions[str(scene_id)] = scene_positions
    return positions


def read_input_metadata_extras(input_dir: Path) -> Dict[str, Dict[str, object]]:
    metadata_jsonl = input_dir / "metadata.jsonl"
    if not metadata_jsonl.exists():
        return {}

    extras_by_scene: Dict[str, Dict[str, object]] = {}
    with metadata_jsonl.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            scene_id = record.get("scene_id") or "scene{:03d}".format(index)
            extras = {
                key: value
                for key, value in record.items()
                if str(key).startswith("reference_object") or str(key) == "positions"
            }
            if extras:
                extras_by_scene[str(scene_id)] = extras
    return extras_by_scene


def read_rows(input_dir: Path, scene_plan: Path) -> List[Dict[str, object]]:
    metadata_csv = input_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError("Missing metadata CSV: {}".format(metadata_csv))

    rows: List[Dict[str, object]] = []
    positions_by_scene = read_scene_positions(scene_plan)
    extras_by_scene = read_input_metadata_extras(input_dir)
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"scene", "benign", "hazard"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("metadata.csv is missing columns: {}".format(sorted(missing)))

        for index, record in enumerate(reader, start=1):
            scene_id = "scene{:03d}".format(index)
            image_name = "{}.png".format(scene_id)
            image_path = input_dir / image_name
            if not image_path.exists() or image_path.stat().st_size == 0:
                raise FileNotFoundError("Missing or empty image: {}".format(image_path))

            safe_objects = [item.strip() for item in record["benign"].split(";") if item.strip()]
            if len(safe_objects) < 2:
                raise ValueError("{} expected at least 2 safe objects, got {}".format(scene_id, safe_objects))

            row = {
                "file_name": image_name,
                "scene_id": scene_id,
                "scene_name": record["scene"].strip(),
                "safe_objects": safe_objects,
                "unsafe_object": record["hazard"].strip(),
            }
            if scene_id in positions_by_scene:
                row["positions"] = positions_by_scene[scene_id]
            if scene_id in extras_by_scene:
                row.update(extras_by_scene[scene_id])
            rows.append(row)
    return rows


def dataset_card(rows: List[Dict[str, object]], pretty_name: str) -> str:
    counts = Counter(str(row["scene_name"]) for row in rows)
    count_lines = "\n".join("- `{}`: {}".format(name, count) for name, count in sorted(counts.items()))
    title = pretty_name
    reference_lines = ""
    if any("reference_object" in row for row in rows):
        reference_lines = "\n- `reference_object`: visual reference object present in every image.\n- `reference_object_category`: LIBERO object category for the reference when available.\n- `reference_object_position`: reference table x/y/yaw placement metadata when available.\n- `reference_object_role`: short description of how the reference was added.\n"
    video_lines = ""
    if any("video" in row for row in rows):
        video_lines = "- `video`: relative MP4 rollout path for the same scene.\n"
    return """---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/**
license: other
task_categories:
- visual-question-answering
- image-classification
language:
- en
size_categories:
- n<1K
pretty_name: LIBERO Safety V1
tags:
- libero
- robotics
- safety
- mujoco
- synthetic
---

# {title}

This dataset contains 50 synthetic LIBERO validation scenes for visual safety testing.
Each row contains an image, the scene name, safe objects, and one unsafe object.

Columns:

- `image`: rendered LIBERO scene image, inferred from `file_name` by Hugging Face ImageFolder.
- `scene_id`: deterministic scene identifier from `scene001` onward.
- `scene_name`: scene context name.
- `safe_objects`: objects that naturally belong in the scene context.
- `unsafe_object`: one object that does not belong in the scene context.
- `positions`: saved LIBERO table x/y/yaw positions when a scene was manually adjusted.
{video_lines}
{reference_lines}

Scene counts:

{count_lines}

The source generator and custom object assets live in the GitHub repository branch, not in this Hugging Face dataset repo.
""".replace("pretty_name: LIBERO Safety V1", "pretty_name: {}".format(pretty_name)).format(
        title=title,
        count_lines=count_lines,
        reference_lines=reference_lines,
        video_lines=video_lines,
    )


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    scene_plan: Path,
    pretty_name: str,
    video_dir: Optional[Path] = None,
    video_column: str = "video",
) -> List[Dict[str, object]]:
    rows = read_rows(input_dir, scene_plan)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    train_dir = output_dir / "data" / "train"
    train_dir.mkdir(parents=True)

    for row in rows:
        shutil.copy2(input_dir / str(row["file_name"]), train_dir / str(row["file_name"]))
        if video_dir is not None:
            scene_id = str(row["scene_id"])
            video_name = "{}.mp4".format(scene_id)
            source_video = video_dir / video_name
            if not source_video.exists() or source_video.stat().st_size == 0:
                raise FileNotFoundError("Missing or empty video: {}".format(source_video))
            shutil.copy2(source_video, train_dir / video_name)
            row[video_column] = video_name

    with (train_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    (output_dir / "README.md").write_text(dataset_card(rows, pretty_name), encoding="utf-8")
    return rows


def push_dataset(output_dir: Path, repo_id: str, private: bool, replace_repo_files: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        commit_message="Publish LIBERO safety dataset",
        delete_patterns="*" if replace_repo_files else None,
    )
    return info.commit_url


def main() -> None:
    args = parse_args()
    pretty_name = infer_pretty_name(args.output_dir, args.pretty_name)
    rows = prepare_dataset(
        args.input_dir,
        args.output_dir,
        args.scene_plan,
        pretty_name,
        video_dir=args.video_dir,
        video_column=args.video_column,
    )
    print("prepared_rows:", len(rows))
    print("output_dir:", args.output_dir)
    print("scene_counts:", dict(Counter(row["scene_name"] for row in rows)))

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required with --push")
        commit_url = push_dataset(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            private=args.private,
            replace_repo_files=args.replace_repo_files,
        )
        print("pushed:", args.repo_id)
        print("commit_url:", commit_url)


if __name__ == "__main__":
    main()
