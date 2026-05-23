# LIBERO Custom Object Extension

This extension demonstrates how to add custom MJCF objects to LIBERO without modifying LIBERO core source files or built-in asset folders.

![Four-object custom scene](docs/assets/four_objects_scene.png)

Run it from the LIBERO repository root:

```bash
conda activate libero
python extensions/libero_custom_object/run.py sample --mode check
python extensions/libero_custom_object/run.py sample --mode viewer
python extensions/libero_custom_object/run.py knife --mode viewer
python extensions/libero_custom_object/run.py scissors --mode viewer
python extensions/libero_custom_object/run.py hammer --mode viewer
python extensions/libero_custom_object/run.py hazards --mode viewer
python extensions/libero_custom_object/run.py steak_knife --mode viewer
python extensions/libero_custom_object/run.py can_opener --mode viewer
python extensions/libero_custom_object/run.py kitchen_hazards --mode viewer
python extensions/libero_custom_object/run.py four_objects --mode viewer
python extensions/libero_custom_object/run.py kitchen_microwave_open --mode check
python extensions/libero_custom_object/run.py kitchen_stove --mode check
python extensions/libero_custom_object/run.py microwave_ball_open --mode check
python extensions/libero_custom_object/run.py microwave_plate_prep --mode check
python extensions/libero_custom_object/run.py kitchen_microwave_open --mode image --output extensions/libero_custom_object/outputs/kitchen_microwave_open.png
python extensions/libero_custom_object/run.py kitchen_stove --mode image --output extensions/libero_custom_object/outputs/kitchen_stove.png
python extensions/libero_custom_object/run.py microwave_ball_open --mode image --camera frontview --camera-distance-scale 1.25 --output extensions/libero_custom_object/outputs/microwave_ball_open.png
python extensions/libero_custom_object/run.py microwave_plate_prep --mode image --camera frontview --camera-distance-scale 1.2 --output extensions/libero_custom_object/outputs/microwave_plate_prep.png
python extensions/libero_custom_object/run_ball_to_microwave.py
python extensions/libero_custom_object/run.py gift_box_hazards --mode check
python extensions/libero_custom_object/run.py gift_box_hazards --mode image --camera frontview --output extensions/libero_custom_object/outputs/gift_box_hazards.png
python extensions/libero_custom_object/generate_safety_scene.py --scene microwave --num-benign 2 --num-dangerous 2 --seed 7 --run-mode image
python extensions/libero_custom_object/generate_safety_scene.py --scene stove --num-benign 3 --num-dangerous 2 --seed 11 --run-mode check
python extensions/libero_custom_object/generate_safety_scene.py --scene microwave --num-benign 1 --num-obvious-dangerous 3 --obvious-style primitive --seed 21 --run-mode check
python extensions/libero_custom_object/generate_safety_scene.py --scene stove --num-benign 1 --num-obvious-dangerous 3 --obvious-style mesh --seed 22 --run-mode image --camera frontview
python extensions/libero_custom_object/generate_scene_dataset.py --count 50 --seed 7 --camera frontview --width 512 --height 512 --output-name scene_dataset_v1
python extensions/libero_custom_object/prepare_hf_scene_dataset.py --repo-id saaduddinM/libero_safety_v1 --push --replace-repo-files
python extensions/libero_custom_object/run_pick_place.py pistol --output extensions/libero_custom_object/outputs/pistol_to_stove.mp4
python extensions/libero_custom_object/run.py two_object --mode image
python extensions/libero_custom_object/run.py path/to/your_task.bddl --mode viewer
```

In viewer mode, close the window, press `q` / `Esc` in the viewer, or press `Ctrl+C` in the terminal to stop the process.

## Asset Storage

This extension stores mesh and texture assets with Git LFS. After cloning the branch, install and pull LFS files before running scenes:

```bash
git lfs install
git lfs pull
```

## Published Dataset Reproduction

The published Hugging Face dataset configs can be recreated locally. For
`v4` and `v5`, use the JSON configs in `configs/` as the source of truth:

- `configs/master_config_v4.json` and `configs/master_config_v5.json` are the
  canonical checked-in configs.
- `configs/current_config_v4.json` and `configs/current_config_v5.json` are
  editable working copies. The web visualizer saves object and camera edits
  here.
- Generation uses `current` by default. Pass `--config-mode master` when you
  want to reproduce the canonical master configs exactly.

### Web UI scene editing

Use the web UI when you want to visually adjust the deterministic v4/v5 scene
layout before regenerating images or videos. The UI is deliberately small: it
loads a local HF ImageFolder dataset, shows the current image/video, exposes the
scene metadata, and lets you edit numeric scene configuration.

What can be edited:

- Object and fixture table placement: `x`, `y`, and `yaw` for each listed
  object in the scene.
- Scene-type camera settings: camera name, zoom/distance scale, and x/y/z
  offsets. Camera edits apply to all scenes of the same scene type.

Buttons:

- `Generate` renders a preview image from the edited values. It writes preview
  files under generated output folders only; it does not change the dataset
  config.
- `Save` writes the edited positions to the active `current_config_*.json`.
  Camera save writes the scene-type camera block to the same current config.

Start both editable visualizers from the repo root:

```bash
python extensions/libero_custom_object/hf_dataset_visualizer.py \
  extensions/libero_custom_object/datasets/libero_safety_v4_upload \
  --host 127.0.0.1 \
  --port 7862 \
  --generator-output-name libero_safety_v4 \
  --config-mode current \
  --video-dir extensions/libero_custom_object/videos/libero_safety_pick_place/v4

python extensions/libero_custom_object/hf_dataset_visualizer.py \
  extensions/libero_custom_object/datasets/libero_safety_v5_upload \
  --host 127.0.0.1 \
  --port 7861 \
  --generator-output-name libero_safety_v5 \
  --config-mode current \
  --video-dir extensions/libero_custom_object/videos/libero_safety_pick_place/v5
```

If you are SSHing into the machine, forward both ports from your local machine:

```bash
ssh -N -L 7861:127.0.0.1:7861 -L 7862:127.0.0.1:7862 rbr-saad@rbr-saad-XPS-8960
```

Then open `http://127.0.0.1:7861` for v5 and `http://127.0.0.1:7862` for v4.

After edits are approved, copy the working configs into master:

```bash
cp extensions/libero_custom_object/configs/current_config_v4.json \
   extensions/libero_custom_object/configs/master_config_v4.json
cp extensions/libero_custom_object/configs/current_config_v5.json \
   extensions/libero_custom_object/configs/master_config_v5.json
```

The image generation path uses the same config fields:

- `scenes[].benign`, `scenes[].dangerous`, and `scenes[].positions` define the
  BDDL object choices and placements.
- `camera.scene_types` defines the rendered image/video camera by scene type.
- `video` defines rollout FPS, record cadence, codec, H.264 profile, pixel
  format, and faststart settings.
- `pick_place` defines which benign object the video script should move.

Rebuild v4/v5 images and Hugging Face upload folders from the current configs:

```bash
python extensions/libero_custom_object/create_v4_v5_specs.py \
  --versions v4 v5 \
  --force \
  --config-mode current

python extensions/libero_custom_object/build_published_datasets.py \
  --versions v4 v5 \
  --config-mode current
```

After edits are promoted to master, use `--config-mode master` in the same
commands for the canonical rebuild.

Regenerate v4/v5 pick-place videos from the same configs:

```bash
python extensions/libero_custom_object/generate_pick_place_videos.py \
  --versions v4 v5 \
  --config-mode current \
  --force \
  --stop-on-error
```

The video writer uses H.264 Baseline with `yuv420p` so the MP4s work in common
browsers and players.
When v4/v5 videos exist under `videos/libero_safety_pick_place/`, the Hugging
Face upload folders include the MP4 files and a `video` metadata column.

To upload the rebuilt configs to the existing Hugging Face dataset repo:

```bash
python extensions/libero_custom_object/build_published_datasets.py \
  --versions v4 v5 \
  --config-mode master \
  --push-hf \
  --repo-id saaduddinM/libero_safety_v1
```

To discard local edits and reset a working config:

```bash
cp extensions/libero_custom_object/configs/master_config_v5.json \
   extensions/libero_custom_object/configs/current_config_v5.json
```

Do not hand-edit generated BDDL, PNGs, MP4s, or Hugging Face upload folders.
Edit the config through the visualizer or JSON, then regenerate.

The older configs `v1`, `v2`, and `v3` can still be recreated from committed
BDDL and metadata specs:

```bash
python extensions/libero_custom_object/build_published_datasets.py \
  --versions v1 v2 v3 \
  --seed 7 \
  --camera frontview \
  --width 512 \
  --height 512 \
  --camera-distance-scale 1.15
```

See `docs/published_dataset_generation.md` for the full fresh-machine setup,
output paths, smoke checks, and optional Hugging Face push command.

## Layout

- `libero_custom_object/assets/objects/alphabet_soup/` is a physical copy of `libero/libero/assets/stable_hope_objects/alphabet_soup/`.
- `libero_custom_object/assets/objects/scissors/`, `hammer/`, and `can_opener/` are selected scanned-object assets from `kevinzakka/mujoco_scanned_objects`.
- `libero_custom_object/assets/objects/knife/` and `steak_knife/` are lightweight primitive MJCF demo objects.
- `libero_custom_object/assets/objects/gift_box/`, `gift_box_lid/`, and `balls/` are lightweight props for fixed hazard-context checks.
- `libero_custom_object/assets/objects/toy_props/` contains primitive toy car, block, ball, train, drum, and ring props. The dataset generator uses the drum instead of the ring.
- `libero_custom_object/assets/objects/robocasa/` contains a curated RoboCasa safety-object subset.
- `libero_custom_object/assets/objects/obvious_hazards/` contains inert cartoon bomb, dynamite, and toy-blaster props for obvious visual hazard tests.
- `libero_custom_object/assets/objects/reference_white_cutting_board_flat*/` contains static non-colliding MuJoCo reference fixtures used by published datasets `v3` and `v5`.
- `libero_custom_object/assets/manifest.yaml` declares custom object categories.
- `libero_custom_object/bddl_files/` contains the sample floor-to-basket BDDL task.
- `libero_custom_object/registry.py` registers manifest objects into LIBERO at runtime.
- `run.py` runs any BDDL config in `viewer`, `image`, or `check` mode.
- `run_ball_to_microwave.py` records a scripted pick-and-place rollout that moves the ball into the open microwave.
- `generate_safety_scene.py` creates randomized kitchen safety-scene BDDL files from the curated RoboCasa object subset and can run them immediately.
- `generate_scene_dataset.py` creates the 50-scene gift-box/stove/microwave validation dataset with BDDL, PNG renders, JSONL metadata, and optional Hugging Face `Dataset.save_to_disk`.
- `build_published_datasets.py` recreates the published Hugging Face configs. `v1`-`v3` use committed BDDL and metadata specs; `v4` and `v5` use the JSON configs in `configs/`.
- `prepare_hf_scene_dataset.py` converts generated PNGs and metadata into Hugging Face ImageFolder layout and can replace the HF dataset repo with dataset-only files.
- `check_kitchen_microwave_open_on_table.bddl` and `check_kitchen_stove_on_table.bddl` are fixture-only kitchen appliance scene checks.
- `check_open_microwave_ball_on_table.bddl` is a simple wall scene with an open microwave and one ball.
- `check_open_microwave_plate_prep.bddl` is a simple wall scene with an open microwave and a plate staged in front of it.

The sample object is registered as `custom_alphabet_soup`, so BDDL files can use:

```lisp
custom_alphabet_soup_1 - custom_alphabet_soup
```

The primitive demo knife is registered as `custom_knife`, so BDDL files can use:

```lisp
custom_knife_1 - custom_knife
```

The scanned dangerous-tool objects are registered as:

```lisp
custom_scissors_1 - custom_scissors
custom_hammer_1 - custom_hammer
custom_steak_knife_1 - custom_steak_knife
custom_can_opener_1 - custom_can_opener
```

The curated RoboCasa imports use `custom_rc_*` category names. See:

- `docs/robocasa_selected_objects.csv` for the selected object index.
- `docs/robocasa_safety_candidates.md` for the larger category-level safety shortlist.

Examples:

```lisp
custom_rc_knife_ov_knife_0_1 - custom_rc_knife_ov_knife_0
custom_rc_tomato_ov_tomato_0_1 - custom_rc_tomato_ov_tomato_0
custom_rc_olive_oil_bottle_ag_olive_oil_bottle_0_1 - custom_rc_olive_oil_bottle_ag_olive_oil_bottle_0
```

The obvious hazard props use `custom_obvious_*` category names. They are
nonfunctional visual props for model-evaluation scenes only. See:

- `docs/obvious_hazard_objects.csv` for the primitive and mesh-backed object index.
- `scripts/import_obvious_hazard_assets.py` for the importer/conversion helper.

Examples:

```lisp
custom_obvious_cartoon_bomb_1 - custom_obvious_cartoon_bomb
custom_obvious_mesh_dynamite_bundle_1 - custom_obvious_mesh_dynamite_bundle
custom_obvious_mesh_toy_gun_1 - custom_obvious_mesh_toy_gun
```

The fixed gift-box hazard scene reuses the TurboSquid dynamite, black bomb,
pistol, and tomato from the fixed stove scene, replaces the bullet with
`ball_1 - custom_baseball_ball`, and leaves `gift_box_lid_1 - custom_gift_box_lid`
on a table corner:

```bash
python extensions/libero_custom_object/run.py gift_box_hazards --mode image --camera frontview
```

## Random Safety Scene Generator

All commands below should be run from the LIBERO repository root after activating the LIBERO environment:

```bash
conda activate libero
```

Generate and run a microwave scene with two benign and two dangerous objects:

```bash
python extensions/libero_custom_object/generate_safety_scene.py \
  --scene microwave \
  --num-benign 2 \
  --num-dangerous 2 \
  --seed 7 \
  --run-mode image \
  --camera frontview
```

Generate and reset-check a stove scene:

```bash
python extensions/libero_custom_object/generate_safety_scene.py \
  --scene stove \
  --num-benign 3 \
  --num-dangerous 2 \
  --seed 11 \
  --run-mode check
```

Generate a microwave scene with obvious primitive hazards:

```bash
python extensions/libero_custom_object/generate_safety_scene.py \
  --scene microwave \
  --num-benign 1 \
  --num-obvious-dangerous 3 \
  --obvious-style primitive \
  --seed 21 \
  --run-mode check
```

Generate a stove render with mesh-backed obvious hazards:

```bash
python extensions/libero_custom_object/generate_safety_scene.py \
  --scene stove \
  --num-benign 1 \
  --num-obvious-dangerous 3 \
  --obvious-style mesh \
  --seed 22 \
  --run-mode image \
  --camera frontview
```

The generator writes:

- a BDDL scene under `generated_bddl/`
- a same-name JSON metadata file listing object labels, categories, hazard group, asset style, and attribution fields
- a PNG under `outputs/` when `--run-mode image` is used

Use `--run-mode none` to only generate the BDDL and metadata.

Run a generated or hand-written scene directly:

```bash
python extensions/libero_custom_object/run.py \
  extensions/libero_custom_object/generated_bddl/<scene_name>.bddl \
  --mode image \
  --camera frontview
```

## Three-Scene Dataset Generator

The dataset generator creates exactly 50 validation scenes by default:

- `scene001`-`scene017`: gift box with two benign toys and one obvious/comic hazard; the hazard slot is varied across scenes
- `scene018`-`scene034`: plain-wall stove scene with the stove fixed on the table and all selected objects placed beside it at fixed scene033/scene034-style triangle centers, never on it; only the frying pan and moka pot are benign
- `scene035`-`scene050`: open microwave fixed on the left with a fixed plate in front, plus two safe food/container objects and one metal/tool hazard in a fixed angled right-side line

Generate the default dataset:

```bash
python extensions/libero_custom_object/generate_scene_dataset.py \
  --count 50 \
  --seed 7 \
  --camera frontview \
  --width 512 \
  --height 512 \
  --output-name scene_dataset_v1
```

The generator writes:

- BDDL files under `extensions/libero_custom_object/generated_bddl/scene_dataset_v1/`
- PNG renders under `extensions/libero_custom_object/outputs/scene_dataset_v1/`
- simple metadata beside the PNGs:
  - `extensions/libero_custom_object/outputs/scene_dataset_v1/metadata.jsonl`
  - `extensions/libero_custom_object/outputs/scene_dataset_v1/metadata.csv`

The metadata has only three fields:

- `scene`: `giftbox`, `stove`, or `microwave`
- `benign`: the two benign object names
- `hazard`: the one unsafe object name

Gift-box gun hazards use the TurboSquid pistol mesh, not the primitive toy gun. The toy ring is not used in generated gift-box scenes. Stove benign objects are restricted to the frying pan and moka pot; food, bottles, bowls, plates, soup cans, knives, and cardboard boxes are treated as hazards in stove scenes. The spray-can hazard is excluded because it obstructs the stove view. The microwave plate is fixed scene context and is not listed as one of the two benign objects. Microwave scenes include knife as a hazard option and exclude the flat scissors assets for visibility.

Add `--save-hf` to also write a Hugging Face `Dataset.save_to_disk` artifact under `extensions/libero_custom_object/datasets/scene_dataset_v1/hf_dataset/`.

Install the optional Hugging Face dependency before generation if you need the saved dataset object:

```bash
python -m pip install datasets pillow
# or, from the extension directory:
python -m pip install -e ".[dataset]"
```

Use `--skip-render` for metadata/BDDL-only regeneration and `--no-clean` to keep an existing output directory.

Prepare the Hugging Face ImageFolder upload directory from generated PNGs:

```bash
python extensions/libero_custom_object/prepare_hf_scene_dataset.py
```

Publish the dataset-only Hugging Face repo. This keeps the generated images, metadata, and dataset card on Hugging Face; the source code and raw object assets stay in GitHub:

```bash
python extensions/libero_custom_object/prepare_hf_scene_dataset.py \
  --repo-id saaduddinM/libero_safety_v1 \
  --push \
  --replace-repo-files
```

## Stove Pick-And-Place Video

List available object instances in the fixed TurboSquid stove scene:

```bash
python extensions/libero_custom_object/run_pick_place.py --list-objects
```

Move one object from the TurboSquid stove scene onto the flat stove and save a video:

```bash
python extensions/libero_custom_object/run_pick_place.py pistol \
  --output extensions/libero_custom_object/outputs/pistol_to_stove.mp4
```

The object argument accepts full BDDL instance names or short aliases:

```bash
python extensions/libero_custom_object/run_pick_place.py bullet
python extensions/libero_custom_object/run_pick_place.py dynamite
python extensions/libero_custom_object/run_pick_place.py bomb
python extensions/libero_custom_object/run_pick_place.py tomato
```

By default the script uses lower object-specific grasp points plus a light object-attachment assist after the gripper closes, which keeps thin mesh objects from slipping during the scripted motion. Add `--physics-only` to test pure gripper physics, or tune `--grasp-fraction` if a mesh still looks too high or too low.

Use another BDDL scene with the same kind of table/stove setup:

```bash
python extensions/libero_custom_object/run_pick_place.py dynamite \
  --bddl extensions/libero_custom_object/libero_custom_object/bddl_files/check_stove_turbosquid_dynamite_bomb_tomato.bddl \
  --output extensions/libero_custom_object/outputs/dynamite_to_stove.mp4
```

## Python Usage

```python
from libero_custom_object import register_custom_objects, get_sample_bddl_path

register_custom_objects()
bddl_file = get_sample_bddl_path()
```

For the two-object sample:

```python
from libero_custom_object import register_custom_objects, get_two_object_bddl_path

register_custom_objects()
bddl_file = get_two_object_bddl_path()
```

For the knife sample:

```python
from libero_custom_object import register_custom_objects, get_knife_bddl_path

register_custom_objects()
bddl_file = get_knife_bddl_path()
```

For the combined scissors-and-hammer sample:

```python
from libero_custom_object import register_custom_objects, get_hazard_tools_bddl_path

register_custom_objects()
bddl_file = get_hazard_tools_bddl_path()
```

For the combined steak-knife-and-can-opener sample:

```python
from libero_custom_object import register_custom_objects, get_kitchen_hazards_bddl_path

register_custom_objects()
bddl_file = get_kitchen_hazards_bddl_path()
```

For the four-object mixed scene:

```python
from libero_custom_object import register_custom_objects, get_four_objects_bddl_path

register_custom_objects()
bddl_file = get_four_objects_bddl_path()
```

For the kitchen appliance scene checks:

```python
from libero_custom_object import (
    get_kitchen_microwave_open_bddl_path,
    get_kitchen_stove_bddl_path,
)

microwave_bddl_file = get_kitchen_microwave_open_bddl_path()
stove_bddl_file = get_kitchen_stove_bddl_path()
```

## Adding A Future Custom Object

1. Add a new MJCF object folder under `libero_custom_object/assets/objects/<asset_name>/`.
2. Ensure the folder contains a MuJoCo XML file and any referenced mesh or texture files.
3. Add an entry to `libero_custom_object/assets/manifest.yaml`:

```yaml
objects:
  - category_name: custom_my_object
    asset_name: my_object
    xml: my_object.xml
    source_path: external/or/original/source/path
    rotation: [0.0, 0.0]
    rotation_axis: x
    description: Short note about the object.
```

Use a `custom_` prefix for category names to avoid collisions with LIBERO's built-in object names.

This extension creates simulation tasks only. It does not include demonstration trajectories; collect those later with LIBERO's existing data collection scripts.

See `THIRD_PARTY_ASSETS.md` for scanned-object asset attribution and licensing notes.
