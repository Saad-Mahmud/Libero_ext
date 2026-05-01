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
python extensions/libero_custom_object/run.py two_object --mode image
python extensions/libero_custom_object/run.py path/to/your_task.bddl --mode viewer
```

In viewer mode, close the window, press `q` / `Esc` in the viewer, or press `Ctrl+C` in the terminal to stop the process.

## Layout

- `libero_custom_object/assets/objects/alphabet_soup/` is a physical copy of `libero/libero/assets/stable_hope_objects/alphabet_soup/`.
- `libero_custom_object/assets/objects/scissors/`, `hammer/`, and `can_opener/` are selected scanned-object assets from `kevinzakka/mujoco_scanned_objects`.
- `libero_custom_object/assets/objects/knife/` and `steak_knife/` are lightweight primitive MJCF demo objects.
- `libero_custom_object/assets/manifest.yaml` declares custom object categories.
- `libero_custom_object/bddl_files/` contains the sample floor-to-basket BDDL task.
- `libero_custom_object/registry.py` registers manifest objects into LIBERO at runtime.
- `run.py` runs any BDDL config in `viewer`, `image`, or `check` mode.

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
