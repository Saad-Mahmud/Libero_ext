# Published Dataset Generation

This document describes how to recreate the published Hugging Face configs in
`saaduddinM/libero_safety_v1` from a fresh clone.

The published configs are:

- `v1`: original 50-scene LIBERO safety image set.
- `v2`: regenerated 50-scene set with the latest deterministic object layout.
- `v3`: same scene objects and labels as `v2`, plus a static non-colliding
  MuJoCo white cutting board reference fixture. The original `v2` objects and
  fixtures are unchanged.
- `v4`: config-driven zoomed non-reference set with pick-place videos.
- `v5`: config-driven zoomed white-cutting-board reference set with pick-place
  videos.

The canonical source files are committed under:

```text
extensions/libero_custom_object/published_dataset_specs/
```

Each version has 50 BDDL files and one `metadata.jsonl`. Rendered PNGs and
Hugging Face upload folders are generated artifacts and are not tracked by git.

For `v4` and `v5`, the source of truth is:

```text
extensions/libero_custom_object/configs/master_config_v4.json
extensions/libero_custom_object/configs/current_config_v4.json
extensions/libero_custom_object/configs/master_config_v5.json
extensions/libero_custom_object/configs/current_config_v5.json
```

Use `master` for canonical reproduction and `current` for visualizer-edited
working layouts. Generation commands default to `current`.

## Environment

From a fresh checkout:

```bash
git clone git@github.com:Saad-Mahmud/Libero_ext.git
cd Libero_ext
git lfs install
git lfs pull

source /home/rbr-saad/anaconda3/etc/profile.d/conda.sh  # adapt to your conda install
conda activate libero
python -m pip install -e .
python -m pip install -e extensions/libero_custom_object
python -m pip install datasets pillow huggingface_hub
export MUJOCO_GL=egl
```

If your machine has a different conda install, replace the `source` line with
the path to that installation or run `conda activate libero` directly.

## Generate Published Sets

Run this from the repository root:

```bash
python extensions/libero_custom_object/build_published_datasets.py \
  --versions v1 v2 v3 \
  --seed 7 \
  --camera frontview \
  --width 512 \
  --height 512 \
  --camera-distance-scale 1.15
```

This writes:

```text
extensions/libero_custom_object/generated_bddl/libero_safety_v1/
extensions/libero_custom_object/generated_bddl/libero_safety_v2/
extensions/libero_custom_object/generated_bddl/libero_safety_v3/

extensions/libero_custom_object/outputs/libero_safety_v1/
extensions/libero_custom_object/outputs/libero_safety_v2/
extensions/libero_custom_object/outputs/libero_safety_v3/

extensions/libero_custom_object/datasets/libero_safety_v1_upload/
extensions/libero_custom_object/datasets/libero_safety_v2_upload/
extensions/libero_custom_object/datasets/libero_safety_v3_upload/
extensions/libero_custom_object/datasets/libero_safety_combined_upload/
```

The combined upload folder is the local Hugging Face ImageFolder layout for the
repo configs `v1`, `v2`, and `v3`.

To build the config-driven `v4` and `v5` sets from the editable working
configs:

```bash
python extensions/libero_custom_object/create_v4_v5_specs.py \
  --versions v4 v5 \
  --force \
  --config-mode current

python extensions/libero_custom_object/build_published_datasets.py \
  --versions v4 v5 \
  --config-mode current
```

Use `--config-mode master` instead when you want to reproduce the canonical
checked-in configs exactly.

To regenerate the matching pick-place videos:

```bash
python extensions/libero_custom_object/generate_pick_place_videos.py \
  --versions v4 v5 \
  --config-mode current \
  --force \
  --stop-on-error
```

When those videos exist, the v4/v5 Hugging Face upload folders include the MP4
files and a `video` column in `metadata.jsonl`.

## Quick Structural Check

To verify the committed BDDL and metadata specs without rendering all PNGs:

```bash
python extensions/libero_custom_object/build_published_datasets.py \
  --versions v1 v2 v3 \
  --skip-render \
  --skip-prepare
```

To render a single generated scene manually after the full build:

```bash
python extensions/libero_custom_object/run.py \
  extensions/libero_custom_object/generated_bddl/libero_safety_v3/scene001.bddl \
  --mode image \
  --camera frontview \
  --width 512 \
  --height 512 \
  --camera-distance-scale 1.15 \
  --seed 1390851128 \
  --output /tmp/libero_safety_v3_scene001.png
```

## Push to Hugging Face

Only do this when you intend to update the dataset repo:

```bash
huggingface-cli login
python extensions/libero_custom_object/build_published_datasets.py \
  --versions v1 v2 v3 \
  --push-hf \
  --repo-id saaduddinM/libero_safety_v1
```

The push command uploads `libero_safety_combined_upload/` to the same dataset
repo, preserving the per-version configs present in that folder.

## Load the Published Dataset

```python
from datasets import load_dataset

v1 = load_dataset("saaduddinM/libero_safety_v1", "v1", split="train")
v2 = load_dataset("saaduddinM/libero_safety_v1", "v2", split="train")
v3 = load_dataset("saaduddinM/libero_safety_v1", "v3", split="train")
v4 = load_dataset("saaduddinM/libero_safety_v1", "v4", split="train")
v5 = load_dataset("saaduddinM/libero_safety_v1", "v5", split="train")
```
