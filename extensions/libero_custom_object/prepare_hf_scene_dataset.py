#!/usr/bin/env python3
"""Prepare and optionally upload the scene dataset in HF ImageFolder format."""

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List


EXTENSION_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = EXTENSION_ROOT / "outputs" / "scene_dataset_v1"
DEFAULT_OUTPUT_DIR = EXTENSION_ROOT / "datasets" / "libero_safety_v1_upload"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Hugging Face ImageFolder upload folder for scene_dataset_v1."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=None, help="Optional HF dataset repo id, e.g. user/libero_safety_v1.")
    parser.add_argument("--push", action="store_true", help="Upload the prepared folder to Hugging Face.")
    parser.add_argument(
        "--replace-repo-files",
        action="store_true",
        help="When pushing, delete existing remote files before uploading this dataset-only folder.",
    )
    parser.add_argument("--private", action="store_true", help="Create the HF dataset repo as private.")
    return parser.parse_args()


def read_rows(input_dir: Path) -> List[Dict[str, object]]:
    metadata_csv = input_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError("Missing metadata CSV: {}".format(metadata_csv))

    rows: List[Dict[str, object]] = []
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
            if len(safe_objects) != 2:
                raise ValueError("{} expected 2 safe objects, got {}".format(scene_id, safe_objects))

            rows.append(
                {
                    "file_name": image_name,
                    "scene_id": scene_id,
                    "scene_name": record["scene"].strip(),
                    "safe_objects": safe_objects,
                    "unsafe_object": record["hazard"].strip(),
                }
            )
    return rows


def dataset_card(rows: List[Dict[str, object]]) -> str:
    counts = Counter(str(row["scene_name"]) for row in rows)
    count_lines = "\n".join("- `{}`: {}".format(name, count) for name, count in sorted(counts.items()))
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

# LIBERO Safety V1

This dataset contains 50 synthetic LIBERO validation scenes for visual safety testing.
Each row contains an image, the scene name, two safe objects, and one unsafe object.

Columns:

- `image`: rendered LIBERO scene image, inferred from `file_name` by Hugging Face ImageFolder.
- `scene_id`: deterministic scene identifier from `scene001` onward.
- `scene_name`: scene context name.
- `safe_objects`: two objects that naturally belong in the scene context.
- `unsafe_object`: one object that does not belong in the scene context.

Scene counts:

{}

The source generator and custom object assets live in the GitHub repository branch, not in this Hugging Face dataset repo.
""".format(
        count_lines
    )


def prepare_dataset(input_dir: Path, output_dir: Path) -> List[Dict[str, object]]:
    rows = read_rows(input_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    train_dir = output_dir / "data" / "train"
    train_dir.mkdir(parents=True)

    for row in rows:
        shutil.copy2(input_dir / str(row["file_name"]), train_dir / str(row["file_name"]))

    with (train_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    (output_dir / "README.md").write_text(dataset_card(rows), encoding="utf-8")
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
    rows = prepare_dataset(args.input_dir, args.output_dir)
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
