import argparse
import csv
import json
import math
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parent
LIBERO_ROOT = EXTENSION_ROOT.parent.parent
SELECTED_OBJECTS_CSV = EXTENSION_ROOT / "docs" / "robocasa_selected_objects.csv"
OBVIOUS_OBJECTS_CSV = EXTENSION_ROOT / "docs" / "obvious_hazard_objects.csv"
OBJECT_ASSET_ROOT = EXTENSION_ROOT / "libero_custom_object" / "assets" / "objects"
DEFAULT_BDDL_DIR = EXTENSION_ROOT / "generated_bddl"
DEFAULT_OUTPUT_DIR = EXTENSION_ROOT / "outputs"
PLACEMENT_MARGIN = 0.035
REGION_HALF_SIZE = 0.002


APPLIANCE_CONFIGS = {
    "microwave": {
        "fixture_name": "microwave_1",
        "fixture_category": "microwave",
        "region_name": "microwave_init_region",
        "region_target": "kitchen_table",
        "range": (-0.01, 0.34, 0.01, 0.36),
        "yaw": (0.0, 0.0),
        "site_regions": [("top_side", "microwave_1"), ("heating_region", "microwave_1")],
        "default_state": "Open",
        "state_options": {"open": "Open", "closed": "Close"},
        "language": "microwave",
    },
    "stove": {
        "fixture_name": "flat_stove_1",
        "fixture_category": "flat_stove",
        "region_name": "flat_stove_init_region",
        "region_target": "kitchen_table",
        "range": (-0.21, 0.19, -0.19, 0.21),
        "yaw": (0.0, 0.0),
        "site_regions": [("cook_region", "flat_stove_1")],
        "default_state": "Turnoff",
        "state_options": {"off": "Turnoff", "on": "Turnon"},
        "language": "stove",
    },
}

APPLIANCE_BLOCKERS = {
    # These are conservative visual exclusion zones on the kitchen table frame.
    "microwave": [(0.0, 0.35, 0.32), (0.35, -0.10, 0.18)],
    "stove": [(-0.20, 0.20, 0.10)],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a randomized LIBERO kitchen safety scene from the curated "
            "RoboCasa benign/dangerous object subset, then optionally run it."
        )
    )
    parser.add_argument("--scene", choices=sorted(APPLIANCE_CONFIGS), required=True)
    parser.add_argument("--num-benign", "--num-safe", type=int, default=0)
    parser.add_argument("--num-dangerous", "--num-unsafe", type=int, default=0)
    parser.add_argument("--num-obvious-dangerous", type=int, default=0)
    parser.add_argument(
        "--obvious-style",
        choices=["primitive", "mesh", "mixed"],
        default="primitive",
        help="Object style used only by --num-obvious-dangerous.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--appliance-state",
        choices=["default", "open", "closed", "on", "off"],
        default="default",
        help="Microwave supports open/closed. Stove supports on/off.",
    )
    parser.add_argument(
        "--output-bddl",
        default=None,
        help="Output BDDL path. Defaults to generated_bddl/<scene>_b<N>_d<N>_seed<S>.bddl.",
    )
    parser.add_argument(
        "--run-mode",
        choices=["none", "check", "image", "viewer"],
        default="check",
        help="Run the generated scene after writing it.",
    )
    parser.add_argument("--camera", default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--image-output",
        default=None,
        help="PNG path used when --run-mode image. Defaults to outputs/<bddl_stem>.png.",
    )
    return parser.parse_args()


def load_csv(path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def choose_objects(rows, label, count, rng):
    pool = [row for row in rows if row["safety_label"] == label]
    if count > len(pool):
        raise ValueError(
            f"Requested {count} {label} objects, but only {len(pool)} are available."
        )
    return rng.sample(pool, count)


def choose_obvious_objects(rows, style, count, rng):
    pool = [
        row
        for row in rows
        if row["safety_label"] == "dangerous"
        and row.get("hazard_group") == "obvious"
        and (style == "mixed" or row.get("asset_style") == style)
    ]
    if count > len(pool):
        raise ValueError(
            f"Requested {count} obvious dangerous objects with style '{style}', "
            f"but only {len(pool)} are available."
        )
    return rng.sample(pool, count)


def row_asset_xml_path(row):
    asset_dir = OBJECT_ASSET_ROOT / row["asset_name"]
    xml_path = asset_dir / "model.xml"
    if xml_path.exists():
        return xml_path
    matches = sorted(asset_dir.glob("*.xml"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No XML file found for {row['category_name']} in {asset_dir}")


def estimate_object_radius(row):
    xml_path = row_asset_xml_path(row)
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse {xml_path}: {exc}") from exc

    horizontal_radius_site = root.find(".//site[@name='horizontal_radius_site']")
    if horizontal_radius_site is not None:
        pos = [
            float(value)
            for value in horizontal_radius_site.attrib.get("pos", "0.08 0 0").split()
        ]
        if pos:
            return max(pos[0], 0.025)

    reg_bbox = root.find(".//geom[@name='reg_bbox']")
    if reg_bbox is None:
        return 0.08

    pos = [float(value) for value in reg_bbox.attrib.get("pos", "0 0 0").split()]
    size = [float(value) for value in reg_bbox.attrib["size"].split()]
    return max(abs(pos[0]) + size[0], abs(pos[1]) + size[1], 0.025)


def table_candidate_centers():
    # Coordinates are relative to the kitchen table frame used by LIBERO BDDL.
    x_values = [-0.35, -0.20, -0.05, 0.10, 0.25]
    y_values = [-0.32, -0.17, -0.02, 0.13, 0.28]
    return [(x, y) for y in y_values for x in x_values]


def center_is_valid(scene, center, radius, placed):
    for appliance_x, appliance_y, appliance_radius in APPLIANCE_BLOCKERS[scene]:
        if (
            math.hypot(center[0] - appliance_x, center[1] - appliance_y)
            < radius + appliance_radius + PLACEMENT_MARGIN
        ):
            return False

    for placed_center, placed_radius in placed:
        if (
            math.hypot(center[0] - placed_center[0], center[1] - placed_center[1])
            < radius + placed_radius + PLACEMENT_MARGIN
        ):
            return False
    return True


def place_objects(scene, objects, rng):
    candidates = table_candidate_centers()
    items = [
        {
            "row": row,
            "radius": estimate_object_radius(row),
            "source_order": index,
        }
        for index, row in enumerate(objects)
    ]
    items.sort(key=lambda item: (-item["radius"], item["source_order"]))

    candidate_orders = []
    for _ in items:
        shuffled_candidates = list(candidates)
        rng.shuffle(shuffled_candidates)
        candidate_orders.append(shuffled_candidates)

    def search(index, placed, placements):
        if index == len(items):
            return placements
        item = items[index]
        for center in candidate_orders[index]:
            if center_is_valid(scene, center, item["radius"], placed):
                result = search(
                    index + 1,
                    placed + [(center, item["radius"])],
                    placements + [(item["row"], center, item["radius"])],
                )
                if result is not None:
                    return result
        return None

    placements = search(0, [], [])
    if placements is None:
        raise ValueError(
            "Could not find non-overlapping table placement for "
            f"{len(objects)} total objects in the {scene} scene. "
            "Try fewer objects or a different seed."
        )
    return placements


def format_region(name, target, rect, yaw):
    x1, y1, x2, y2 = rect
    yaw1, yaw2 = yaw
    return f"""      ({name}
          (:target {target})
          (:ranges (
              ({x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f})
            )
          )
          (:yaw_rotation (
              ({yaw1:.6f} {yaw2:.6f})
            )
          )
      )"""


def resolve_appliance_state(scene, requested_state):
    config = APPLIANCE_CONFIGS[scene]
    if requested_state == "default":
        return config["default_state"]
    if requested_state not in config["state_options"]:
        valid = ", ".join(["default"] + sorted(config["state_options"]))
        raise ValueError(f"{scene} does not support state '{requested_state}'. Use: {valid}.")
    return config["state_options"][requested_state]


def default_bddl_path(
    scene,
    num_benign,
    num_dangerous,
    num_obvious_dangerous,
    obvious_style,
    seed,
):
    seed_part = "random" if seed is None else str(seed)
    if num_obvious_dangerous:
        name = (
            f"{scene}_safety_scene_b{num_benign}_d{num_dangerous}_"
            f"o{num_obvious_dangerous}_{obvious_style}_seed{seed_part}.bddl"
        )
    else:
        name = (
            f"{scene}_safety_scene_b{num_benign}_d{num_dangerous}_"
            f"seed{seed_part}.bddl"
        )
    return DEFAULT_BDDL_DIR / name


def build_bddl(scene, placements, appliance_state, rng):
    config = APPLIANCE_CONFIGS[scene]
    object_entries = []
    regions = [
        format_region(
            config["region_name"],
            config["region_target"],
            config["range"],
            config["yaw"],
        )
    ]

    for site_region_name, target in config["site_regions"]:
        regions.append(f"""      ({site_region_name}
          (:target {target})
      )""")

    for index, (row, (x, y), radius) in enumerate(placements, start=1):
        region_name = f"object_{index}_init_region"
        yaw = rng.uniform(-math.pi, math.pi)
        regions.append(
            format_region(
                region_name,
                "kitchen_table",
                (
                    x - REGION_HALF_SIZE,
                    y - REGION_HALF_SIZE,
                    x + REGION_HALF_SIZE,
                    y + REGION_HALF_SIZE,
                ),
                (yaw, yaw),
            )
        )
        object_entries.append(
            {
                "instance_name": f"{row['category_name']}_{index}",
                "category_name": row["category_name"],
                "region_name": f"kitchen_table_{region_name}",
                "safety_label": row["safety_label"],
                "robocasa_category": row["category"],
                "robocasa_model": row["model"],
                "robocasa_source": row["source"],
                "hazard_group": row.get("hazard_group", "kitchen"),
                "asset_style": row.get("asset_style", "robocasa"),
                "source_url": row.get("source_url", ""),
                "license": row.get("license", ""),
                "attribution": row.get("attribution", ""),
                "download_status": row.get("download_status", ""),
                "table_xy": [x, y],
                "estimated_radius": radius,
            }
        )

    language_parts = [
        f"check a randomized {config['language']} safety scene",
        f"with {sum(o['safety_label'] == 'benign' for o in object_entries)} benign objects",
        f"and {sum(o['safety_label'] == 'dangerous' for o in object_entries)} dangerous objects",
    ]
    obvious_count = sum(
        o["safety_label"] == "dangerous" and o.get("hazard_group") == "obvious"
        for o in object_entries
    )
    if obvious_count:
        language_parts.append(f"including {obvious_count} obvious visual hazards")

    object_lines = "\n".join(
        f"    {entry['instance_name']} - {entry['category_name']}"
        for entry in object_entries
    )
    interest_lines = "\n".join(
        ["    " + config["fixture_name"]]
        + [f"    {entry['instance_name']}" for entry in object_entries]
    )
    init_lines = "\n".join(
        [
            f"    (On {config['fixture_name']} kitchen_table_{config['region_name']})",
            f"    ({appliance_state} {config['fixture_name']})",
        ]
        + [
            f"    (On {entry['instance_name']} {entry['region_name']})"
            for entry in object_entries
        ]
    )
    goal_lines = "\n".join(
        [f"      ({appliance_state} {config['fixture_name']})"]
        + [
            f"      (On {entry['instance_name']} {entry['region_name']})"
            for entry in object_entries
        ]
    )

    bddl = f"""(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {' '.join(language_parts)})
    (:regions
{chr(10).join(regions)}
    )

  (:fixtures
    kitchen_table - kitchen_table
    {config['fixture_name']} - {config['fixture_category']}
  )

  (:objects
{object_lines}
  )

  (:obj_of_interest
{interest_lines}
  )

  (:init
{init_lines}
  )

  (:goal
    (And
{goal_lines}
    )
  )

)
"""
    return bddl, object_entries


def write_metadata(path, scene, seed, appliance_state, objects):
    metadata = {
        "scene": scene,
        "seed": seed,
        "appliance_state": appliance_state,
        "objects": objects,
    }
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata_path


def run_scene(args, bddl_path):
    if args.run_mode == "none":
        return None

    cmd = [
        sys.executable,
        str(EXTENSION_ROOT / "run.py"),
        str(bddl_path),
        "--mode",
        args.run_mode,
    ]
    if args.camera is not None:
        cmd.extend(["--camera", args.camera])
    if args.run_mode == "image":
        output = (
            Path(args.image_output)
            if args.image_output is not None
            else DEFAULT_OUTPUT_DIR / f"{bddl_path.stem}.png"
        )
        if not output.is_absolute():
            output = LIBERO_ROOT / output
        cmd.extend(
            [
                "--output",
                str(output),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
            ]
        )

    print("running:", " ".join(str(part) for part in cmd))
    return subprocess.run(cmd, cwd=LIBERO_ROOT, check=True)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    rows = load_csv(SELECTED_OBJECTS_CSV)
    obvious_rows = load_csv(OBVIOUS_OBJECTS_CSV)
    selected = (
        choose_objects(rows, "benign", args.num_benign, rng)
        + choose_objects(rows, "dangerous", args.num_dangerous, rng)
        + choose_obvious_objects(
            obvious_rows,
            args.obvious_style,
            args.num_obvious_dangerous,
            rng,
        )
    )
    rng.shuffle(selected)
    placements = place_objects(args.scene, selected, rng)
    appliance_state = resolve_appliance_state(args.scene, args.appliance_state)

    bddl_path = (
        Path(args.output_bddl)
        if args.output_bddl is not None
        else default_bddl_path(
            args.scene,
            args.num_benign,
            args.num_dangerous,
            args.num_obvious_dangerous,
            args.obvious_style,
            args.seed,
        )
    )
    if not bddl_path.is_absolute():
        bddl_path = LIBERO_ROOT / bddl_path
    bddl_path.parent.mkdir(parents=True, exist_ok=True)

    bddl, object_entries = build_bddl(args.scene, placements, appliance_state, rng)
    bddl_path.write_text(bddl)
    metadata_path = write_metadata(
        bddl_path,
        args.scene,
        args.seed,
        appliance_state,
        object_entries,
    )

    print("generated_bddl:", bddl_path)
    print("generated_metadata:", metadata_path)
    print("objects:")
    for entry in object_entries:
        print(
            f"  {entry['safety_label']:9s} {entry['instance_name']} "
            f"({entry['robocasa_category']}/{entry['robocasa_model']}, "
            f"{entry['hazard_group']}/{entry['asset_style']})"
        )

    run_scene(args, bddl_path)


if __name__ == "__main__":
    main()
