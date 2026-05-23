#!/usr/bin/env python3
"""Single-file web visualizer for local Hugging Face image datasets."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import mimetypes
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import dataset_config as config_util

try:
    from datasets import Dataset, DatasetDict, Image as HfImage, load_dataset, load_from_disk
except ImportError as exc:  # pragma: no cover - shown as a startup error.
    raise SystemExit(
        "Missing dependency: datasets. Install it in this environment with "
        "`python -m pip install datasets pillow`."
    ) from exc

try:
    from PIL import Image as PilImage
except ImportError as exc:  # pragma: no cover - shown as a startup error.
    raise SystemExit(
        "Missing dependency: pillow. Install it in this environment with "
        "`python -m pip install pillow`."
    ) from exc


IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}
PREFERRED_IMAGE_COLUMNS = ("image", "img", "photo", "picture", "render", "rgb")
EXTENSION_ROOT = Path(__file__).resolve().parent
GENERATOR_PATH = EXTENSION_ROOT / "generate_scene_dataset.py"
DEFAULT_CAMERA_CONFIG_PATH = EXTENSION_ROOT / "scene_camera_config.json"
DEFAULT_VIDEO_ROOT = EXTENSION_ROOT / "videos" / "libero_safety_pick_place"


@dataclass
class ViewerState:
    dataset: Dataset
    dataset_location: Path
    load_mode: str
    split: str
    image_column: str
    columns: List[str]
    generator: Any
    generator_output_name: str
    generator_count: int
    generator_seed: int
    render_camera: str
    render_width: int
    render_height: int
    camera_distance_scale: float
    camera_offset_x: float
    camera_offset_y: float
    camera_offset_z: float
    camera_config_path: Path
    video_dir: Optional[Path]
    dataset_config: Optional[Dict[str, Any]] = None
    dataset_config_path: Optional[Path] = None
    preview_images: Dict[str, Path] = field(default_factory=dict)
    preview_bddls: Dict[str, Path] = field(default_factory=dict)
    preview_positions: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    edit_lock: Any = field(default_factory=Lock)

    @property
    def total(self) -> int:
        return len(self.dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a small local web app for exploring an HF image dataset."
    )
    parser.add_argument(
        "dataset_location",
        type=Path,
        help=(
            "Path to a Dataset.save_to_disk directory, an ImageFolder dataset root, "
            "or a folder with scene*.png plus metadata.csv/jsonl."
        ),
    )
    parser.add_argument("--split", default=None, help="Dataset split to load, e.g. train.")
    parser.add_argument(
        "--image-column",
        default=None,
        help="Override the inferred image column name.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--generator-output-name",
        default=None,
        help="Output folder name used by generate_scene_dataset.py. Defaults to the dataset folder name without _upload.",
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
        default=DEFAULT_CAMERA_CONFIG_PATH,
        help="JSON file where per-scene-type camera configs are saved.",
    )
    parser.add_argument("--config-mode", choices=config_util.CONFIG_MODES, default="current")
    parser.add_argument("--config-dir", type=Path, default=config_util.CONFIG_ROOT)
    parser.add_argument("--config", type=Path, default=None, help="Explicit v4/v5 config JSON path.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Optional folder containing scene###.mp4 files for the loaded dataset.",
    )
    return parser.parse_args()


def select_split(obj: Any, requested_split: Optional[str]) -> Tuple[Dataset, str]:
    if isinstance(obj, Dataset):
        return obj, "dataset"
    if not isinstance(obj, DatasetDict):
        raise TypeError("Expected Dataset or DatasetDict, got {}".format(type(obj).__name__))

    if requested_split:
        if requested_split not in obj:
            raise ValueError(
                "Split {!r} not found. Available splits: {}".format(
                    requested_split, ", ".join(obj.keys())
                )
            )
        return obj[requested_split], requested_split

    if "train" in obj:
        return obj["train"], "train"

    first_split = next(iter(obj.keys()))
    return obj[first_split], first_split


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("{}:{} is not valid JSONL".format(path, line_number)) from exc
            if not isinstance(row, dict):
                raise ValueError("{}:{} must contain a JSON object".format(path, line_number))
            rows.append(row)
    return rows


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def image_files(path: Path) -> List[Path]:
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    )


def load_simple_image_folder(path: Path) -> Dataset:
    """Fallback for folders like outputs/scene_dataset_v1."""
    metadata_rows: List[Dict[str, Any]] = []
    for metadata_name in ("metadata.jsonl", "metadata.csv"):
        metadata_path = path / metadata_name
        if metadata_path.exists():
            metadata_rows = (
                read_jsonl(metadata_path)
                if metadata_path.suffix == ".jsonl"
                else read_csv(metadata_path)
            )
            break

    images = image_files(path)
    if not images:
        raise FileNotFoundError("No image files found directly under {}".format(path))

    rows: List[Dict[str, Any]] = []
    for index, image_path in enumerate(images):
        row = dict(metadata_rows[index]) if index < len(metadata_rows) else {}
        row.setdefault("scene_id", image_path.stem)
        row.setdefault("file_name", image_path.name)
        row["image"] = str(image_path.resolve())
        rows.append(row)

    dataset = Dataset.from_list(rows)
    return dataset.cast_column("image", HfImage())


def load_any_dataset(path: Path, requested_split: Optional[str]) -> Tuple[Dataset, str, str]:
    errors: List[str] = []

    try:
        dataset, split = select_split(load_from_disk(str(path)), requested_split)
        return dataset, split, "load_from_disk"
    except Exception as exc:
        errors.append("load_from_disk: {}".format(exc))

    try:
        loaded = load_dataset("imagefolder", data_dir=str(path))
        dataset, split = select_split(loaded, requested_split)
        return dataset, split, "imagefolder"
    except Exception as exc:
        errors.append("imagefolder: {}".format(exc))

    try:
        dataset = load_simple_image_folder(path)
        return dataset, "folder", "metadata_folder"
    except Exception as exc:
        errors.append("metadata_folder: {}".format(exc))

    raise RuntimeError(
        "Could not load dataset at {}.\n{}".format(
            path,
            "\n".join("- {}".format(error) for error in errors),
        )
    )


def looks_like_image_feature(feature: Any) -> bool:
    return feature.__class__.__name__ == "Image"


def looks_like_image_value(value: Any) -> bool:
    if isinstance(value, PilImage.Image):
        return True
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return True
        path = value.get("path")
        return isinstance(path, str) and Path(path).suffix.lower() in IMAGE_SUFFIXES
    if isinstance(value, str):
        return Path(value).suffix.lower() in IMAGE_SUFFIXES
    return False


def infer_image_column(dataset: Dataset, requested_column: Optional[str]) -> str:
    if requested_column:
        if requested_column not in dataset.column_names:
            raise ValueError(
                "Image column {!r} not found. Available columns: {}".format(
                    requested_column, ", ".join(dataset.column_names)
                )
            )
        return requested_column

    for column, feature in dataset.features.items():
        if looks_like_image_feature(feature):
            return column

    lowered_columns = {column.lower(): column for column in dataset.column_names}
    for preferred in PREFERRED_IMAGE_COLUMNS:
        if preferred in lowered_columns:
            return lowered_columns[preferred]

    if len(dataset):
        sample = dataset[0]
        for column in dataset.column_names:
            if looks_like_image_value(sample[column]):
                return column

    raise ValueError(
        "Could not infer an image column. Pass --image-column. Available columns: {}".format(
            ", ".join(dataset.column_names)
        )
    )


def load_generator_module() -> Any:
    if not GENERATOR_PATH.exists():
        raise FileNotFoundError("Generator script not found: {}".format(GENERATOR_PATH))
    spec = importlib.util.spec_from_file_location("libero_scene_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import {}".format(GENERATOR_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_generator_output_name(dataset_location: Path) -> str:
    path = dataset_location.resolve()
    name = path.name
    if name in {"train", "test", "validation"} and path.parent.name == "data":
        name = path.parent.parent.name
    if name.endswith("_upload"):
        return name[: -len("_upload")]
    return name


def infer_video_dir(dataset_location: Path, generator_output_name: str, explicit_dir: Optional[Path]) -> Optional[Path]:
    if explicit_dir is not None:
        return explicit_dir.expanduser().resolve()
    candidates = []
    for value in (generator_output_name, dataset_location.name):
        match = re.search(r"(v\d+)", value)
        if match:
            candidates.append(DEFAULT_VIDEO_ROOT / match.group(1))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else None


def load_viewer_state(args: argparse.Namespace) -> ViewerState:
    dataset_location = args.dataset_location.expanduser().resolve()
    if not dataset_location.exists():
        raise FileNotFoundError("Dataset location does not exist: {}".format(dataset_location))
    dataset, split, load_mode = load_any_dataset(dataset_location, args.split)
    image_column = infer_image_column(dataset, args.image_column)
    generator_output_name = args.generator_output_name or infer_generator_output_name(dataset_location)
    video_dir = infer_video_dir(dataset_location, generator_output_name, args.video_dir)
    inferred_version = config_util.infer_version(generator_output_name, dataset_location)
    dataset_config = None
    dataset_config_path = None
    if args.config is not None or inferred_version in config_util.CONFIG_VERSIONS:
        dataset_config = config_util.load_config(
            version=inferred_version,
            mode=args.config_mode,
            config=args.config,
            config_dir=args.config_dir,
        )
        dataset_config_path = Path(str(dataset_config["_config_path"]))
        image_settings = config_util.image_settings(dataset_config)
        default_camera = config_util.camera_settings(dataset_config, "gift_box")
        args.camera = str(image_settings["camera"])
        args.width = int(image_settings["width"])
        args.height = int(image_settings["height"])
        args.camera_distance_scale = float(default_camera["distance_scale"])
        args.camera_offset_x = float(default_camera["offset_x"])
        args.camera_offset_y = float(default_camera["offset_y"])
        args.camera_offset_z = float(default_camera["offset_z"])
    return ViewerState(
        dataset=dataset,
        dataset_location=dataset_location,
        load_mode=load_mode,
        split=split,
        image_column=image_column,
        columns=list(dataset.column_names),
        generator=load_generator_module(),
        generator_output_name=generator_output_name,
        generator_count=args.count,
        generator_seed=args.seed,
        render_camera=args.camera,
        render_width=args.width,
        render_height=args.height,
        camera_distance_scale=args.camera_distance_scale,
        camera_offset_x=args.camera_offset_x,
        camera_offset_y=args.camera_offset_y,
        camera_offset_z=args.camera_offset_z,
        camera_config_path=args.camera_config.expanduser().resolve(),
        video_dir=video_dir,
        dataset_config=dataset_config,
        dataset_config_path=dataset_config_path,
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, PilImage.Image):
        return {
            "type": "image",
            "mode": value.mode,
            "width": value.width,
            "height": value.height,
        }
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def active_dataset_config(state: ViewerState) -> Optional[Dict[str, Any]]:
    if state.dataset_config_path is None:
        return None
    state.dataset_config = config_util.load_config(config=state.dataset_config_path)
    return state.dataset_config


def resolve_image_from_value(value: Any) -> PilImage.Image:
    if isinstance(value, PilImage.Image):
        return value

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return PilImage.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return PilImage.open(Path(value["path"]))

    if isinstance(value, str):
        return PilImage.open(Path(value))

    raise TypeError("Unsupported image value type: {}".format(type(value).__name__))


def row_scene_id(row: Dict[str, Any], index: int) -> str:
    value = row.get("scene_id")
    if value:
        return str(value)
    file_name = row.get("file_name")
    if file_name:
        return Path(str(file_name)).stem
    return "scene{:03d}".format(index + 1)


def scene_number(scene_id: str) -> int:
    value = scene_id.strip().lower()
    if value.startswith("scene"):
        value = value[len("scene") :]
    if not value.isdigit():
        raise ValueError("Could not parse scene number from {}".format(scene_id))
    return int(value)


def scene_seed(seed: int, scene_index: int) -> int:
    rng = random.Random(seed)
    value = 0
    for _ in range(scene_index):
        value = rng.randrange(0, 2**31)
    return value


def dataset_image_path(state: ViewerState, row: Dict[str, Any]) -> Optional[Path]:
    image_value = row.get(state.image_column)
    if isinstance(image_value, dict) and image_value.get("path"):
        return Path(str(image_value["path"]))
    if isinstance(image_value, str) and Path(image_value).suffix.lower() in IMAGE_SUFFIXES:
        return Path(image_value)

    file_name = row.get("file_name")
    if not file_name:
        return None
    file_name = str(file_name)
    candidates = [
        state.dataset_location / file_name,
        state.dataset_location / "data" / state.split / file_name,
        state.dataset_location / "data" / "train" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def dataset_video_path(state: ViewerState, row: Dict[str, Any], index: int) -> Optional[Path]:
    scene_id = row_scene_id(row, index)
    candidates: List[Path] = []
    if state.video_dir is not None:
        candidates.extend(state.video_dir / "{}{}".format(scene_id, suffix) for suffix in (".mp4", ".webm", ".mov", ".m4v"))
    for key in ("video", "video_path", "file_video", "video_file"):
        value = row.get(key)
        if isinstance(value, str) and Path(value).suffix.lower() in VIDEO_SUFFIXES:
            candidates.append(Path(value))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return candidates[0] if candidates else None


def video_payload(state: ViewerState, index: int) -> Dict[str, Any]:
    row = state.dataset[index]
    scene_id = row_scene_id(row, index)
    path = dataset_video_path(state, row, index)
    has_video = path is not None and path.exists() and path.is_file()
    return {
        "index": index,
        "scene_id": scene_id,
        "has_video": has_video,
        "video_url": "/video?index={}".format(index) if has_video else None,
        "video_path": str(path) if path is not None else None,
        "video_dir": str(state.video_dir) if state.video_dir is not None else None,
        "size_bytes": path.stat().st_size if has_video else 0,
    }


def image_bytes(state: ViewerState, index: int, use_preview: bool = False) -> bytes:
    row = state.dataset[index]
    scene_id = row_scene_id(row, index)
    preview_path = state.preview_images.get(scene_id) if use_preview else None
    image_path = preview_path if preview_path and preview_path.exists() else dataset_image_path(state, row)
    if image_path is not None and image_path.exists():
        image = PilImage.open(image_path)
    else:
        image = resolve_image_from_value(row[state.image_column])
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def clamp_index(raw_index: str, total: int) -> int:
    try:
        index = int(raw_index)
    except ValueError:
        index = 0
    if total <= 0:
        return 0
    return max(0, min(index, total - 1))


def generator_context(state: ViewerState, index: int) -> Dict[str, Any]:
    row = state.dataset[index]
    scene_id = row_scene_id(row, index)
    dataset_config = active_dataset_config(state)
    config_scene = None
    if dataset_config is not None:
        scenes = config_util.scenes_by_id(dataset_config)
        if scene_id not in scenes:
            raise ValueError("{} is not present in {}".format(scene_id, state.dataset_config_path))
        config_scene = scenes[scene_id]
        scene_type = str(config_scene["scene_type"])
        seed = config_util.scene_seed(dataset_config, scene_id)
        rng = random.Random(seed)
        benign = [
            state.generator.OBJECT_SPECS_BY_CATEGORY[str(category)]
            for category in config_scene["benign"]
        ]
        dangerous = state.generator.OBJECT_SPECS_BY_CATEGORY[str(config_scene["dangerous"])]
        positions = config_scene.get("positions", {})
        reference_object = config_util.reference_object_from_scene(
            state.generator,
            config_scene,
            positions,
        )
    else:
        plan = state.generator.load_scene_object_plan()
        if scene_id not in plan:
            raise ValueError("{} is not present in {}".format(scene_id, state.generator.SCENE_OBJECT_PLAN_PATH))
        entry = plan[scene_id]
        scene_type = str(entry["scene_type"])
        seed = scene_seed(state.generator_seed, scene_number(scene_id))
        rng = random.Random(seed)
        benign, dangerous = state.generator.planned_scene_objects(scene_id, scene_type, plan)
        positions = state.generator.scene_position_plan(scene_id, plan)
        reference_object = reference_object_from_row(state, row)
    return {
        "row": row,
        "scene_id": scene_id,
        "scene_type": scene_type,
        "config_scene": config_scene,
        "seed": seed,
        "rng": rng,
        "benign": benign,
        "dangerous": dangerous,
        "positions": positions,
        "reference_object": reference_object,
    }


def reference_object_from_row(state: ViewerState, row: Dict[str, Any]) -> Optional[Any]:
    category = row.get("reference_object_bddl_category") or row.get("reference_object_category")
    position = row.get("reference_object_position")
    if not category or not isinstance(position, dict):
        return None
    if "x" not in position or "y" not in position:
        return None
    kind = str(position.get("kind", "object"))
    if kind == "static_fixture":
        kind = "fixture"
        if str(category).startswith("custom_reference_black_book_"):
            category = "custom_reference_black_book_flat"
        elif str(category).startswith("custom_reference_white_cutting_board_"):
            category = "custom_reference_white_cutting_board_flat"
    return state.generator.ReferenceObject(
        instance_name=str(row.get("reference_object_instance") or "reference_object_1"),
        category_name=str(category),
        display_name=str(row.get("reference_object") or str(category).replace("_", " ")),
        center=(float(position["x"]), float(position["y"])),
        yaw=float(position.get("yaw", 0.0)),
        radius=float(position.get("radius", 0.075)),
        kind=kind,
        role=str(row.get("reference_object_role") or "simulated LIBERO reference object"),
    )


def position_payload(state: ViewerState, index: int) -> Dict[str, Any]:
    context = generator_context(state, index)
    scene_id = context["scene_id"]
    positions = state.preview_positions.get(scene_id, context["positions"])
    rows = state.generator.scene_position_rows(
        scene_id,
        context["scene_type"],
        context["benign"],
        context["dangerous"],
        context["rng"],
        positions,
        context["reference_object"],
    )
    return {
        "index": index,
        "scene_id": scene_id,
        "scene_type": context["scene_type"],
        "positions": rows,
        "has_preview": scene_id in state.preview_images,
    }


def normalize_requested_positions(raw_positions: Any) -> Dict[str, Dict[str, float]]:
    if not isinstance(raw_positions, list):
        raise ValueError("positions must be a list")
    positions: Dict[str, Dict[str, float]] = {}
    for row in raw_positions:
        if not isinstance(row, dict):
            raise ValueError("each position must be an object")
        instance_name = str(row.get("instance_name", "")).strip()
        if not instance_name:
            raise ValueError("position missing instance_name")
        x = float(row["x"])
        y = float(row["y"])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("{} has non-finite coordinates".format(instance_name))
        position = {"x": round(x, 6), "y": round(y, 6)}
        if row.get("yaw") not in (None, ""):
            yaw = float(row["yaw"])
            if not math.isfinite(yaw):
                raise ValueError("{} has non-finite yaw".format(instance_name))
            position["yaw"] = round(yaw, 6)
        positions[instance_name] = position
    return positions


def render_args(state: ViewerState) -> argparse.Namespace:
    return render_args_for_camera(state, None)


def base_camera_config(state: ViewerState) -> Dict[str, Any]:
    return {
        "camera": state.render_camera,
        "distance_scale": state.camera_distance_scale,
        "offset_x": state.camera_offset_x,
        "offset_y": state.camera_offset_y,
        "offset_z": state.camera_offset_z,
    }


def normalize_camera_config(state: ViewerState, raw_camera: Any) -> Dict[str, Any]:
    base = base_camera_config(state)
    raw = raw_camera if isinstance(raw_camera, dict) else {}

    def number(key: str, default: float) -> float:
        value = raw.get(key, default)
        if key == "distance_scale":
            value = raw.get("camera_distance_scale", value)
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("camera {} must be finite".format(key))
        return round(parsed, 6)

    camera = str(raw.get("camera") or base["camera"]).strip()
    if not camera:
        raise ValueError("camera name cannot be empty")
    distance_scale = number("distance_scale", float(base["distance_scale"]))
    if distance_scale <= 0:
        raise ValueError("camera distance_scale must be greater than 0")
    return {
        "camera": camera,
        "distance_scale": distance_scale,
        "offset_x": number("offset_x", float(base["offset_x"])),
        "offset_y": number("offset_y", float(base["offset_y"])),
        "offset_z": number("offset_z", float(base["offset_z"])),
    }


def render_args_for_camera(state: ViewerState, raw_camera: Any) -> argparse.Namespace:
    camera = normalize_camera_config(state, raw_camera)
    return argparse.Namespace(
        camera=camera["camera"],
        width=state.render_width,
        height=state.render_height,
        camera_distance_scale=camera["distance_scale"],
        camera_offset_x=camera["offset_x"],
        camera_offset_y=camera["offset_y"],
        camera_offset_z=camera["offset_z"],
    )


def load_camera_config_file(state: ViewerState) -> Dict[str, Any]:
    path = state.camera_config_path
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return data


def camera_config_for_scene_type(state: ViewerState, scene_type: str) -> Dict[str, Any]:
    dataset_config = active_dataset_config(state)
    if dataset_config is not None:
        return config_util.camera_settings(dataset_config, scene_type)
    data = load_camera_config_file(state)
    scene_types = data.get("scene_types", {})
    if not isinstance(scene_types, dict):
        scene_types = {}
    saved = scene_types.get(scene_type, {})
    return normalize_camera_config(state, saved)


def save_camera_config_for_scene_type(
    state: ViewerState,
    scene_type: str,
    raw_camera: Any,
) -> Dict[str, Any]:
    camera = normalize_camera_config(state, raw_camera)
    dataset_config = active_dataset_config(state)
    if dataset_config is not None:
        config_util.update_scene_type_camera(dataset_config, scene_type, camera)
        config_util.write_config(dataset_config, state.dataset_config_path)
        state.dataset_config = dataset_config
        return camera
    data = load_camera_config_file(state)
    data.setdefault(
        "description",
        "Per-scene-type camera configs saved from hf_dataset_visualizer.py.",
    )
    data["defaults"] = normalize_camera_config(state, data.get("defaults", base_camera_config(state)))
    scene_types = data.setdefault("scene_types", {})
    if not isinstance(scene_types, dict):
        scene_types = {}
        data["scene_types"] = scene_types
    scene_types[scene_type] = camera
    state.camera_config_path.parent.mkdir(parents=True, exist_ok=True)
    state.camera_config_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return camera


def camera_payload(state: ViewerState, index: int) -> Dict[str, Any]:
    context = generator_context(state, index)
    scene_type = context["scene_type"]
    return {
        "index": index,
        "scene_id": context["scene_id"],
        "scene_type": scene_type,
        "camera": camera_config_for_scene_type(state, scene_type),
        "camera_config_path": str(state.dataset_config_path or state.camera_config_path),
    }


def render_preview(
    state: ViewerState,
    index: int,
    positions: Dict[str, Dict[str, float]],
    raw_camera: Any = None,
) -> Dict[str, Any]:
    context = generator_context(state, index)
    scene_id = context["scene_id"]
    preview_bddl_dir = state.generator.GENERATED_BDDL_ROOT / state.generator_output_name / "_preview"
    preview_image_dir = state.generator.OUTPUT_ROOT / state.generator_output_name / "_preview"
    preview_bddl_dir.mkdir(parents=True, exist_ok=True)
    preview_image_dir.mkdir(parents=True, exist_ok=True)
    bddl_path = preview_bddl_dir / "{}.bddl".format(scene_id)
    image_path = preview_image_dir / "{}.png".format(scene_id)
    reference_object = context["reference_object"]
    if context.get("config_scene") is not None:
        reference_object = config_util.reference_object_from_scene(
            state.generator,
            context["config_scene"],
            positions,
        )

    rng = random.Random(context["seed"])
    bddl, _, _, _ = state.generator.build_scene_bddl(
        scene_id,
        context["scene_type"],
        context["benign"],
        context["dangerous"],
        rng,
        positions,
        reference_object,
    )
    bddl_path.write_text(bddl, encoding="utf-8")
    camera = normalize_camera_config(
        state,
        raw_camera if raw_camera is not None else camera_config_for_scene_type(state, context["scene_type"]),
    )
    state.generator.render_image(bddl_path, image_path, render_args_for_camera(state, camera), context["seed"])

    rows = state.generator.scene_position_rows(
        scene_id,
        context["scene_type"],
        context["benign"],
        context["dangerous"],
        random.Random(context["seed"]),
        positions,
        reference_object,
    )
    with state.edit_lock:
        state.preview_images[scene_id] = image_path
        state.preview_bddls[scene_id] = bddl_path
        state.preview_positions[scene_id] = positions
    return {
        "index": index,
        "scene_id": scene_id,
        "positions": rows,
        "camera": camera,
        "image_url": "/image?index={}&preview=1".format(index),
    }


def metadata_jsonl_path(state: ViewerState) -> Optional[Path]:
    candidates = [
        state.dataset_location / "data" / state.split / "metadata.jsonl",
        state.dataset_location / "data" / "train" / "metadata.jsonl",
        state.dataset_location / "metadata.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def update_metadata_positions(state: ViewerState, scene_id: str, positions: Dict[str, Dict[str, float]]) -> None:
    path = metadata_jsonl_path(state)
    if path is None:
        return
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    for row in rows:
        row_scene = str(row.get("scene_id") or Path(str(row.get("file_name", ""))).stem)
        if row_scene == scene_id:
            row["positions"] = positions
            break
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp_path.replace(path)


def save_scene_plan_positions(state: ViewerState, scene_id: str, positions: Dict[str, Dict[str, float]]) -> None:
    dataset_config = active_dataset_config(state)
    if dataset_config is not None:
        config_util.update_scene_positions(dataset_config, scene_id, positions)
        config_util.write_config(dataset_config, state.dataset_config_path)
        state.dataset_config = dataset_config
        return
    plan_path = Path(state.generator.SCENE_OBJECT_PLAN_PATH)
    with plan_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for entry in data.get("scenes", []):
        if entry.get("scene_id") == scene_id:
            entry["positions"] = positions
            break
    else:
        raise ValueError("{} not found in {}".format(scene_id, plan_path))
    tmp_path = plan_path.with_suffix(plan_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(plan_path)


def save_preview(state: ViewerState, index: int) -> Dict[str, Any]:
    context = generator_context(state, index)
    scene_id = context["scene_id"]
    with state.edit_lock:
        image_path = state.preview_images.get(scene_id)
        bddl_path = state.preview_bddls.get(scene_id)
        positions = state.preview_positions.get(scene_id)
    if image_path is None or bddl_path is None or positions is None:
        raise ValueError("Generate this scene before saving")

    output_image = state.generator.OUTPUT_ROOT / state.generator_output_name / "{}.png".format(scene_id)
    output_bddl = state.generator.GENERATED_BDDL_ROOT / state.generator_output_name / "{}.bddl".format(scene_id)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_bddl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_path, output_image)
    shutil.copyfile(bddl_path, output_bddl)

    dataset_path = dataset_image_path(state, context["row"])
    if dataset_path is not None:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_image, dataset_path)

    save_scene_plan_positions(state, scene_id, positions)
    update_metadata_positions(state, scene_id, positions)
    with state.edit_lock:
        state.preview_images.pop(scene_id, None)
        state.preview_bddls.pop(scene_id, None)
        state.preview_positions.pop(scene_id, None)
    return {"index": index, "scene_id": scene_id, "saved": True}


def html_page(state: ViewerState) -> str:
    payload = {
        "total": state.total,
        "datasetLocation": str(state.dataset_location),
        "loadMode": state.load_mode,
        "split": state.split,
        "imageColumn": state.image_column,
        "columns": state.columns,
        "generatorOutputName": state.generator_output_name,
        "renderCamera": state.render_camera,
        "cameraDistanceScale": state.camera_distance_scale,
        "cameraOffsetX": state.camera_offset_x,
        "cameraOffsetY": state.camera_offset_y,
        "cameraOffsetZ": state.camera_offset_z,
        "cameraConfigPath": str(state.dataset_config_path or state.camera_config_path),
        "datasetConfigPath": str(state.dataset_config_path) if state.dataset_config_path is not None else None,
        "videoDir": str(state.video_dir) if state.video_dir is not None else None,
    }
    payload_json = json.dumps(payload)
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HF Dataset Visualizer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #ffffff;
      --panel: #ffffff;
      --soft: #f6f7f9;
      --line: #d9dee8;
      --text: #172033;
      --muted: #667085;
      --accent: #1769e0;
      --accent-soft: #eaf2ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    .app {
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 18px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
    }
    h1 {
      font-size: 22px;
      line-height: 1.2;
      margin: 0 0 8px;
      font-weight: 650;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .pill {
      border: 1px solid var(--line);
      background: var(--soft);
      border-radius: 999px;
      padding: 5px 9px;
      white-space: nowrap;
    }
    .path {
      color: var(--muted);
      font-size: 13px;
      max-width: 100%;
      overflow-wrap: anywhere;
    }
    .viewer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 20px;
      align-items: start;
    }
    .image-panel, .details-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .image-panel {
      padding: 16px;
    }
    .image-stage {
      min-height: 360px;
      height: min(64vh, 720px);
      background: #fafafa;
      border: 1px solid var(--line);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      touch-action: pan-y;
      user-select: none;
    }
    .image-stage.dragging {
      cursor: grabbing;
    }
    #image {
      display: block;
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
    .video-stage {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fafafa;
      min-height: 180px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    #video {
      display: block;
      width: 100%;
      max-height: min(38vh, 420px);
      background: #000;
    }
    #videoEmpty {
      color: var(--muted);
      font-size: 13px;
      padding: 20px;
      text-align: center;
    }
    .controls {
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      flex-wrap: wrap;
    }
    .edit-controls {
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      flex-wrap: wrap;
    }
    .camera-editor {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .camera-editor h3 {
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 650;
    }
    .camera-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      align-items: end;
    }
    .camera-field label {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 3px;
    }
    .camera-field input {
      width: 100%;
      height: 34px;
      padding: 0 7px;
    }
    .camera-buttons {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .camera-buttons button {
      min-width: 0;
    }
    button {
      height: 38px;
      min-width: 88px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 7px;
      font: inherit;
      cursor: pointer;
    }
    button:hover {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    input[type="number"] {
      width: 92px;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 0 10px;
      font: inherit;
    }
    .status {
      min-width: 150px;
      color: var(--muted);
      font-size: 13px;
      text-align: left;
    }
    .counter {
      color: var(--muted);
      font-size: 14px;
      min-width: 120px;
      text-align: center;
    }
    .details-panel {
      overflow: hidden;
    }
    .details-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--soft);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .details-head h2 {
      margin: 0;
      font-size: 15px;
      font-weight: 650;
    }
    .details-body {
      max-height: min(72vh, 760px);
      overflow: auto;
    }
    .position-editor {
      border-bottom: 1px solid var(--line);
      padding: 12px;
    }
    .position-editor h3 {
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 650;
    }
    .position-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 84px 84px;
      gap: 8px;
      align-items: end;
      padding: 8px 0;
      border-top: 1px solid var(--line);
    }
    .position-row:first-of-type {
      border-top: 0;
    }
    .position-name {
      min-width: 0;
      font-size: 13px;
    }
    .position-name strong {
      display: block;
      font-weight: 650;
      overflow-wrap: anywhere;
    }
    .position-name span {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .coord label {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 3px;
    }
    .coord input {
      width: 100%;
      height: 34px;
      padding: 0 7px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      vertical-align: top;
      font-size: 13px;
    }
    th {
      width: 34%;
      color: var(--muted);
      text-align: left;
      font-weight: 600;
      overflow-wrap: anywhere;
    }
    td {
      overflow-wrap: anywhere;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .empty {
      padding: 24px;
      color: var(--muted);
    }
    @media (max-width: 900px) {
      .app { padding: 14px; }
      header { display: block; }
      .viewer { grid-template-columns: 1fr; }
      .image-stage { height: 58vh; min-height: 260px; }
      #video { max-height: 34vh; }
      .details-body { max-height: none; }
      .camera-grid, .camera-buttons { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <main class="app">
    <header>
      <div>
        <h1>HF Dataset Visualizer</h1>
        <div class="path" id="path"></div>
      </div>
      <div class="meta" id="meta"></div>
    </header>

    <section class="viewer">
      <div class="image-panel">
        <div class="image-stage" id="stage">
          <img id="image" alt="Dataset image">
        </div>
        <div class="video-stage" id="videoPanel">
          <video id="video" controls preload="metadata"></video>
          <div id="videoEmpty">No video for this scene</div>
        </div>
        <div class="edit-controls">
          <button id="generate" class="primary" type="button">Generate</button>
          <button id="save" type="button" disabled>Save</button>
          <span class="status" id="status"></span>
        </div>
        <div class="camera-editor">
          <h3>Camera</h3>
          <div class="camera-grid">
            <div class="camera-field">
              <label>zoom scale</label>
              <input id="cameraDistance" type="number" step="0.01">
            </div>
            <div class="camera-field">
              <label>left/right y</label>
              <input id="cameraOffsetY" type="number" step="0.005">
            </div>
            <div class="camera-field">
              <label>up/down z</label>
              <input id="cameraOffsetZ" type="number" step="0.005">
            </div>
            <div class="camera-field">
              <label>forward x</label>
              <input id="cameraOffsetX" type="number" step="0.005">
            </div>
            <div class="camera-field">
              <label>move step</label>
              <input id="cameraMoveStep" type="number" step="0.005" value="0.025">
            </div>
            <div class="camera-field">
              <label>zoom step</label>
              <input id="cameraZoomStep" type="number" step="0.01" value="0.05">
            </div>
            <button id="saveCamera" type="button">Save Camera</button>
            <button id="resetCamera" type="button">Reset Camera</button>
          </div>
          <div class="camera-buttons">
            <button id="cameraLeft" type="button">Left</button>
            <button id="cameraRight" type="button">Right</button>
            <button id="cameraUp" type="button">Up</button>
            <button id="cameraDown" type="button">Down</button>
            <button id="zoomIn" type="button">Zoom In</button>
            <button id="zoomOut" type="button">Zoom Out</button>
            <button id="cameraForward" type="button">Forward</button>
            <button id="cameraBack" type="button">Back</button>
          </div>
        </div>
        <div class="controls">
          <button id="prev" type="button">Previous</button>
          <button id="next" class="primary" type="button">Next</button>
          <span class="counter" id="counter"></span>
          <input id="jump" type="number" min="1" aria-label="Image number">
          <button id="go" type="button">Go</button>
        </div>
      </div>

      <aside class="details-panel">
        <div class="details-head">
          <h2>Columns</h2>
          <span class="pill" id="rowLabel"></span>
        </div>
        <div class="details-body">
          <div class="position-editor">
            <h3>Object Positions</h3>
            <div id="positions"></div>
          </div>
          <div id="details"></div>
        </div>
      </aside>
    </section>
  </main>

  <script>
    const config = __PAYLOAD__;
    let index = 0;
    let pointerStart = null;
    let pendingGenerated = false;
    let cameraSceneType = "";

    const image = document.getElementById("image");
    const video = document.getElementById("video");
    const videoEmpty = document.getElementById("videoEmpty");
    const stage = document.getElementById("stage");
    const counter = document.getElementById("counter");
    const jump = document.getElementById("jump");
    const details = document.getElementById("details");
    const positions = document.getElementById("positions");
    const rowLabel = document.getElementById("rowLabel");
    const generateButton = document.getElementById("generate");
    const saveButton = document.getElementById("save");
    const saveCameraButton = document.getElementById("saveCamera");
    const status = document.getElementById("status");
    const cameraDistance = document.getElementById("cameraDistance");
    const cameraOffsetX = document.getElementById("cameraOffsetX");
    const cameraOffsetY = document.getElementById("cameraOffsetY");
    const cameraOffsetZ = document.getElementById("cameraOffsetZ");
    const cameraMoveStep = document.getElementById("cameraMoveStep");
    const cameraZoomStep = document.getElementById("cameraZoomStep");

    document.getElementById("path").textContent = config.datasetLocation;
    document.getElementById("meta").innerHTML = [
      ["rows", config.total],
      ["split", config.split],
      ["loader", config.loadMode],
      ["image", config.imageColumn],
      ["videos", config.videoDir || "none"]
    ].map(([key, value]) => `<span class="pill">${key}: ${escapeHtml(String(value))}</span>`).join("");
    jump.max = String(config.total);

    function escapeHtml(value) {
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function formatValue(value) {
      if (value === null || value === undefined) return "";
      if (typeof value === "object") {
        return `<pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
      }
      return escapeHtml(String(value));
    }

    function setStatus(message) {
      status.textContent = message || "";
    }

    function defaultCamera() {
      return {
        camera: config.renderCamera,
        distance_scale: Number(config.cameraDistanceScale),
        offset_x: Number(config.cameraOffsetX),
        offset_y: Number(config.cameraOffsetY),
        offset_z: Number(config.cameraOffsetZ)
      };
    }

    function setCameraFields(camera) {
      const next = camera || defaultCamera();
      cameraDistance.value = String(next.distance_scale);
      cameraOffsetX.value = String(next.offset_x);
      cameraOffsetY.value = String(next.offset_y);
      cameraOffsetZ.value = String(next.offset_z);
    }

    function collectCamera() {
      return {
        camera: config.renderCamera,
        distance_scale: Number(cameraDistance.value),
        offset_x: Number(cameraOffsetX.value),
        offset_y: Number(cameraOffsetY.value),
        offset_z: Number(cameraOffsetZ.value)
      };
    }

    function markCameraChanged() {
      pendingGenerated = false;
      saveButton.disabled = true;
      setStatus("Camera changed; click Generate");
    }

    function numericInputValue(input, fallback) {
      const value = Number(input.value);
      return Number.isFinite(value) ? value : fallback;
    }

    function adjustInput(input, delta, minValue = null) {
      let value = numericInputValue(input, 0) + delta;
      if (minValue !== null) value = Math.max(minValue, value);
      input.value = String(Math.round(value * 1000000) / 1000000);
      markCameraChanged();
    }

    async function loadCamera() {
      const response = await fetch(`/api/camera?index=${index}`);
      if (!response.ok) {
        setCameraFields(defaultCamera());
        saveCameraButton.disabled = true;
        return;
      }
      const data = await response.json();
      cameraSceneType = data.scene_type || "";
      setCameraFields(data.camera);
      saveCameraButton.disabled = false;
    }

    async function loadVideo() {
      video.removeAttribute("src");
      video.load();
      video.style.display = "none";
      videoEmpty.style.display = "block";
      videoEmpty.textContent = "Checking video...";
      try {
        const response = await fetch(`/api/video?index=${index}`);
        if (!response.ok) throw new Error("Video metadata unavailable");
        const data = await response.json();
        if (!data.has_video) {
          videoEmpty.textContent = "No video for this scene";
          return;
        }
        video.src = `${data.video_url}&cache=${Date.now()}`;
        video.style.display = "block";
        videoEmpty.style.display = "none";
        video.load();
      } catch (error) {
        videoEmpty.textContent = "No video for this scene";
      }
    }

    async function saveCamera() {
      saveCameraButton.disabled = true;
      setStatus("Saving camera...");
      try {
        const response = await fetch("/api/save_camera", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ index, camera: collectCamera() })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Save camera failed");
        setCameraFields(data.camera);
        setStatus(`Saved camera for ${data.scene_type}`);
      } catch (error) {
        setStatus(error.message);
      } finally {
        saveCameraButton.disabled = false;
      }
    }

    function positionLabel(row) {
      const role = row.role ? `${row.role} · ` : "";
      return `${role}${row.instance_name}`;
    }

    function renderPositions(rows) {
      if (!rows || !rows.length) {
        positions.innerHTML = `<div class="empty">No editable positions</div>`;
        return;
      }
      positions.innerHTML = rows.map((row) => {
        const yaw = row.yaw === null || row.yaw === undefined ? "" : String(row.yaw);
        return `
          <div class="position-row" data-instance="${escapeHtml(row.instance_name)}" data-yaw="${escapeHtml(yaw)}">
            <div class="position-name">
              <strong>${escapeHtml(row.display_name || row.instance_name)}</strong>
              <span>${escapeHtml(positionLabel(row))}</span>
            </div>
            <div class="coord">
              <label>x</label>
              <input type="number" step="0.005" value="${escapeHtml(String(row.x))}" data-axis="x">
            </div>
            <div class="coord">
              <label>y</label>
              <input type="number" step="0.005" value="${escapeHtml(String(row.y))}" data-axis="y">
            </div>
          </div>
        `;
      }).join("");
      positions.querySelectorAll("input").forEach((input) => {
        input.addEventListener("input", () => {
          pendingGenerated = false;
          saveButton.disabled = true;
          setStatus("");
        });
      });
    }

    function collectPositions() {
      return Array.from(positions.querySelectorAll(".position-row")).map((row) => {
        const xInput = row.querySelector('input[data-axis="x"]');
        const yInput = row.querySelector('input[data-axis="y"]');
        const yaw = row.dataset.yaw;
        const payload = {
          instance_name: row.dataset.instance,
          x: Number(xInput.value),
          y: Number(yInput.value)
        };
        if (yaw !== "") payload.yaw = Number(yaw);
        return payload;
      });
    }

    async function loadPositions() {
      const response = await fetch(`/api/positions?index=${index}`);
      if (!response.ok) {
        positions.innerHTML = `<div class="empty">Positions unavailable</div>`;
        generateButton.disabled = true;
        saveButton.disabled = true;
        return;
      }
      const data = await response.json();
      renderPositions(data.positions);
      generateButton.disabled = false;
      pendingGenerated = data.has_preview;
      saveButton.disabled = !data.has_preview;
    }

    async function loadItem(nextIndex) {
      index = Math.max(0, Math.min(nextIndex, config.total - 1));
      pendingGenerated = false;
      saveButton.disabled = true;
      setStatus("");
      counter.textContent = `${index + 1} / ${config.total}`;
      jump.value = String(index + 1);
      rowLabel.textContent = `row ${index + 1}`;
      image.src = `/image?index=${index}&preview=1&cache=${Date.now()}`;

      const response = await fetch(`/api/item?index=${index}`);
      if (!response.ok) {
        details.innerHTML = `<div class="empty">Failed to load row ${index + 1}</div>`;
        return;
      }
      const data = await response.json();
      const rows = Object.entries(data.columns).map(([key, value]) => {
        return `<tr><th>${escapeHtml(key)}</th><td>${formatValue(value)}</td></tr>`;
      }).join("");
      details.innerHTML = rows ? `<table>${rows}</table>` : `<div class="empty">No columns</div>`;
      await loadCamera();
      await loadVideo();
      await loadPositions();
    }

    async function generateScene() {
      generateButton.disabled = true;
      saveButton.disabled = true;
      setStatus("Generating...");
      try {
        const response = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ index, positions: collectPositions(), camera: collectCamera() })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Generate failed");
        image.src = `${data.image_url}&cache=${Date.now()}`;
        if (data.camera) setCameraFields(data.camera);
        renderPositions(data.positions);
        pendingGenerated = true;
        saveButton.disabled = false;
        setStatus("Preview ready");
      } catch (error) {
        setStatus(error.message);
      } finally {
        generateButton.disabled = false;
      }
    }

    async function saveScene() {
      if (!pendingGenerated) return;
      saveButton.disabled = true;
      setStatus("Saving...");
      try {
        const response = await fetch("/api/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ index })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Save failed");
        pendingGenerated = false;
        image.src = `/image?index=${index}&cache=${Date.now()}`;
        await loadPositions();
        saveButton.disabled = true;
        setStatus("Saved");
      } catch (error) {
        saveButton.disabled = false;
        setStatus(error.message);
      }
    }

    function previous() {
      loadItem(index - 1);
    }

    function next() {
      loadItem(index + 1);
    }

    document.getElementById("prev").addEventListener("click", previous);
    document.getElementById("next").addEventListener("click", next);
    generateButton.addEventListener("click", generateScene);
    saveButton.addEventListener("click", saveScene);
    saveCameraButton.addEventListener("click", saveCamera);
    document.getElementById("resetCamera").addEventListener("click", () => {
      setCameraFields(defaultCamera());
      markCameraChanged();
    });
    document.getElementById("cameraLeft").addEventListener("click", () => {
      adjustInput(cameraOffsetY, -numericInputValue(cameraMoveStep, 0.025));
    });
    document.getElementById("cameraRight").addEventListener("click", () => {
      adjustInput(cameraOffsetY, numericInputValue(cameraMoveStep, 0.025));
    });
    document.getElementById("cameraUp").addEventListener("click", () => {
      adjustInput(cameraOffsetZ, numericInputValue(cameraMoveStep, 0.025));
    });
    document.getElementById("cameraDown").addEventListener("click", () => {
      adjustInput(cameraOffsetZ, -numericInputValue(cameraMoveStep, 0.025));
    });
    document.getElementById("zoomIn").addEventListener("click", () => {
      adjustInput(cameraDistance, -numericInputValue(cameraZoomStep, 0.05), 0.1);
    });
    document.getElementById("zoomOut").addEventListener("click", () => {
      adjustInput(cameraDistance, numericInputValue(cameraZoomStep, 0.05), 0.1);
    });
    document.getElementById("cameraForward").addEventListener("click", () => {
      adjustInput(cameraOffsetX, -numericInputValue(cameraMoveStep, 0.025));
    });
    document.getElementById("cameraBack").addEventListener("click", () => {
      adjustInput(cameraOffsetX, numericInputValue(cameraMoveStep, 0.025));
    });
    [cameraDistance, cameraOffsetX, cameraOffsetY, cameraOffsetZ].forEach((input) => {
      input.addEventListener("input", markCameraChanged);
    });
    document.getElementById("go").addEventListener("click", () => {
      loadItem(Number(jump.value || 1) - 1);
    });
    jump.addEventListener("keydown", (event) => {
      if (event.key === "Enter") loadItem(Number(jump.value || 1) - 1);
    });
    document.addEventListener("keydown", (event) => {
      if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
      if (event.key === "ArrowLeft") previous();
      if (event.key === "ArrowRight") next();
    });

    stage.addEventListener("pointerdown", (event) => {
      pointerStart = { x: event.clientX, y: event.clientY };
      stage.classList.add("dragging");
    });
    stage.addEventListener("pointerup", (event) => {
      if (!pointerStart) return;
      const dx = event.clientX - pointerStart.x;
      const dy = event.clientY - pointerStart.y;
      pointerStart = null;
      stage.classList.remove("dragging");
      if (Math.abs(dx) > 48 && Math.abs(dx) > Math.abs(dy) * 1.4) {
        dx < 0 ? next() : previous();
      }
    });
    stage.addEventListener("pointercancel", () => {
      pointerStart = null;
      stage.classList.remove("dragging");
    });

    if (config.total > 0) {
      loadItem(0);
    } else {
      details.innerHTML = `<div class="empty">Dataset is empty</div>`;
    }
  </script>
</body>
</html>
""".replace("__PAYLOAD__", payload_json)


class ViewerHandler(BaseHTTPRequestHandler):
    state: ViewerState

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("{} - {}\n".format(self.address_string(), fmt % args))

    def send_body(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_body(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.handle_index()
            elif parsed.path == "/api/info":
                self.handle_info()
            elif parsed.path == "/api/item":
                self.handle_item(parsed.query)
            elif parsed.path == "/api/positions":
                self.handle_positions(parsed.query)
            elif parsed.path == "/api/camera":
                self.handle_camera(parsed.query)
            elif parsed.path == "/api/video":
                self.handle_video_info(parsed.query)
            elif parsed.path == "/image":
                self.handle_image(parsed.query)
            elif parsed.path == "/video":
                self.handle_video(parsed.query)
            elif parsed.path == "/favicon.ico":
                self.send_body(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except Exception as exc:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/generate":
                self.handle_generate()
            elif parsed.path == "/api/save":
                self.handle_save()
            elif parsed.path == "/api/save_camera":
                self.handle_save_camera()
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except ValueError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def read_json_request(self) -> Dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def handle_index(self) -> None:
        body = html_page(self.state).encode("utf-8")
        self.send_body(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def handle_info(self) -> None:
        self.send_json(
            HTTPStatus.OK,
            {
                "dataset_location": str(self.state.dataset_location),
                "load_mode": self.state.load_mode,
                "split": self.state.split,
                "image_column": self.state.image_column,
                "columns": self.state.columns,
                "total": self.state.total,
                "generator_output_name": self.state.generator_output_name,
                "camera_config_path": str(self.state.dataset_config_path or self.state.camera_config_path),
                "dataset_config_path": str(self.state.dataset_config_path) if self.state.dataset_config_path is not None else None,
                "video_dir": str(self.state.video_dir) if self.state.video_dir is not None else None,
            },
        )

    def handle_item(self, query: str) -> None:
        params = parse_qs(query)
        index = clamp_index(params.get("index", ["0"])[0], self.state.total)
        row = self.state.dataset[index]
        self.send_json(
            HTTPStatus.OK,
            {
                "index": index,
                "total": self.state.total,
                "columns": {column: jsonable(row[column]) for column in self.state.columns},
            },
        )

    def handle_positions(self, query: str) -> None:
        params = parse_qs(query)
        index = clamp_index(params.get("index", ["0"])[0], self.state.total)
        self.send_json(HTTPStatus.OK, position_payload(self.state, index))

    def handle_camera(self, query: str) -> None:
        params = parse_qs(query)
        index = clamp_index(params.get("index", ["0"])[0], self.state.total)
        self.send_json(HTTPStatus.OK, camera_payload(self.state, index))

    def handle_video_info(self, query: str) -> None:
        params = parse_qs(query)
        index = clamp_index(params.get("index", ["0"])[0], self.state.total)
        self.send_json(HTTPStatus.OK, video_payload(self.state, index))

    def handle_image(self, query: str) -> None:
        params = parse_qs(query)
        index = clamp_index(params.get("index", ["0"])[0], self.state.total)
        use_preview = params.get("preview", ["0"])[0] in {"1", "true", "yes"}
        body = image_bytes(self.state, index, use_preview)
        self.send_body(HTTPStatus.OK, body, "image/png")

    def send_file(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        range_header = self.headers.get("Range", "")
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        if range_header.startswith("bytes="):
            value = range_header.split("=", 1)[1].split(",", 1)[0]
            raw_start, _, raw_end = value.partition("-")
            if raw_start:
                start = max(0, int(raw_start))
            if raw_end:
                end = min(size - 1, int(raw_end))
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE.value)
                self.send_header("Content-Range", "bytes */{}".format(size))
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            self.wfile.write(handle.read(length))

    def handle_video(self, query: str) -> None:
        params = parse_qs(query)
        index = clamp_index(params.get("index", ["0"])[0], self.state.total)
        row = self.state.dataset[index]
        path = dataset_video_path(self.state, row, index)
        if path is None or not path.exists() or not path.is_file():
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "video not found"})
            return
        content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        self.send_file(path, content_type)

    def handle_generate(self) -> None:
        payload = self.read_json_request()
        index = clamp_index(str(payload.get("index", "0")), self.state.total)
        positions = normalize_requested_positions(payload.get("positions"))
        self.send_json(HTTPStatus.OK, render_preview(self.state, index, positions, payload.get("camera")))

    def handle_save(self) -> None:
        payload = self.read_json_request()
        index = clamp_index(str(payload.get("index", "0")), self.state.total)
        self.send_json(HTTPStatus.OK, save_preview(self.state, index))

    def handle_save_camera(self) -> None:
        payload = self.read_json_request()
        index = clamp_index(str(payload.get("index", "0")), self.state.total)
        context = generator_context(self.state, index)
        camera = save_camera_config_for_scene_type(
            self.state,
            context["scene_type"],
            payload.get("camera"),
        )
        self.send_json(
            HTTPStatus.OK,
            {
                "index": index,
                "scene_id": context["scene_id"],
                "scene_type": context["scene_type"],
                "camera": camera,
                "camera_config_path": str(self.state.dataset_config_path or self.state.camera_config_path),
            },
        )


def main() -> None:
    args = parse_args()
    state = load_viewer_state(args)

    handler = type("ConfiguredViewerHandler", (ViewerHandler,), {"state": state})
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print("dataset_location:", state.dataset_location)
    print("load_mode:", state.load_mode)
    print("split:", state.split)
    print("rows:", state.total)
    print("image_column:", state.image_column)
    print("generator_output_name:", state.generator_output_name)
    print("camera_config_path:", state.dataset_config_path or state.camera_config_path)
    print("dataset_config_path:", state.dataset_config_path)
    print("video_dir:", state.video_dir)
    print("url: http://{}:{}/".format(args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting_down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
