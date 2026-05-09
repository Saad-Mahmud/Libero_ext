# Third-Party Assets

This extension includes selected assets converted from:

- `Diamond_Visions_Scissors_Red`
- `Cole_Hardware_Hammer_Black`
- `OXO_Soft_Works_Can_Opener_SnapLock`

Source repository:

- https://github.com/kevinzakka/mujoco_scanned_objects

Per that repository, the MJCF XML files are MIT licensed and the 3D assets
(`.obj` and `.png`) are CC-BY 4.0 licensed.

If you use these scanned-object assets in research, follow the citation guidance
in the upstream repository README.

This extension also includes a copied `alphabet_soup` sample asset from the
LIBERO repository under `libero/libero/assets/stable_hope_objects/alphabet_soup`.
The surrounding LIBERO repository is MIT licensed.

## RoboCasa Safety Object Subset

This extension includes a curated subset of RoboCasa object assets under:

- `libero_custom_object/assets/objects/robocasa/`

The subset was selected from RoboCasa's Objaverse, AI-generated, and Lightwheel
object registries for kitchen-appliance safety experiments.

Source project:

- https://github.com/robocasa/robocasa
- https://robocasa.ai/docs/build/html/assets/objects.html

Per the RoboCasa repository README, RoboCasa code is MIT licensed and assets /
datasets are CC-BY 4.0 licensed. If you use these assets in research, follow
the citation guidance in the upstream RoboCasa repository.

## Obvious Hazard Props

This extension includes inert obvious-hazard props under:

- `libero_custom_object/assets/objects/obvious_hazards/`

The primitive variants are local MJCF-only visual props generated for this
extension. They do not model any functional mechanism.

The mesh-backed variants use or reference the following sources:

- Cartoon bomb concept: https://pixabay.com/3d-models/bomb-fuse-explosive-danger-black-478/
  - Source page lists the asset under the Pixabay Content License.
  - Automated download was blocked in this environment, so the included mesh-backed bomb uses a generated low-poly OBJ fallback matching the requested cartoon-round-bomb concept.
- Dynamite bundle: https://www.get3dmodels.com/tools-and-gadgets/dynamite-stick-bundle/
  - Source page lists the license as CC Attribution and author as Chenchanchong.
  - The downloaded GLB is converted to OBJ for MuJoCo.
- User-provided TurboSquid dynamite:
  - Source URL provided by the user: https://www.turbosquid.com/3d-models/bombs-collection-3d-model-2110019
  - The local FBX is converted to a normalized OBJ for MuJoCo.
  - Verify TurboSquid redistribution rights before publishing this repository or derived assets.
- User-provided TurboSquid pistol and bullet:
  - The local OBJ files are converted to normalized OBJs for MuJoCo.
  - Verify TurboSquid redistribution rights before publishing this repository or derived assets.
- Toy/sci-fi blaster: https://kenney.nl/assets/blaster-kit
  - Kenney Blaster Kit is CC0 1.0 Universal.
  - The included mesh uses `blaster-a.obj` from the downloaded kit, normalized and wrapped for LIBERO.

These props are for simulation safety-evaluation scenes only. They are visual,
nonfunctional objects and do not include physical construction details.
