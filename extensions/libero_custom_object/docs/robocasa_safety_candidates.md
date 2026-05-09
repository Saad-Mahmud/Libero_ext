# RoboCasa Safety Object Candidates

Source catalog: https://robocasa.ai/docs/build/html/assets/objects.html
Source registry: https://github.com/robocasa/robocasa/blob/main/robocasa/models/objects/kitchen_objects.py

RoboCasa exposes 198 object categories in the current registry table. The total documented model instances across these categories is 3134.

## Recommended Benign Stove Objects

These are food-like categories suitable for tasks where the safe action is to place the object on a stove/cook region.

| Category | Groups | Instances | Microwavable | Cookable | Sources |
|---|---:|---:|---:|---:|---|
| `potato` | vegetable | 32 | True | True | objaverse aigen |
| `tomato` | vegetable | 17 | True | True | objaverse aigen |
| `carrot` | vegetable | 16 | True | True | objaverse aigen |
| `bell_pepper` | vegetable | 11 | True | True | objaverse aigen |
| `broccoli` | vegetable | 18 | True | True | objaverse aigen |
| `corn` | vegetable | 12 | True | True | objaverse aigen |
| `mushroom` | vegetable | 23 | True | True | objaverse aigen |
| `onion` | vegetable | 16 | True | True | objaverse aigen |
| `eggplant` | vegetable | 12 | True | True | objaverse aigen |
| `zucchini` | vegetable | 11 | True | True | aigen |
| `fish` | meat | 20 | True | True | objaverse aigen |
| `steak` | meat | 21 | True | True | objaverse aigen |
| `chicken_breast` | meat | 12 | True | True | aigen |
| `sausage` | meat | 16 | True | True | aigen |
| `egg` | dairy | 28 | True | True | objaverse aigen |
| `cheese` | dairy | 22 | True | True | objaverse aigen |
| `tofu` | misc | 17 | False | True | aigen |
| `hot_dog` | cooked_food | 18 | True | True | objaverse aigen |
| `skewers` | meat cooked_food | 6 | True | True | aigen |
| `asparagus` | vegetable | 13 | True | True | aigen |

## Recommended Benign Microwave/Oven Objects

Use these for safe microwave/oven-like placement. In LIBERO's first pass, oven-like means the articulated microwave.

| Category | Groups | Instances | Microwavable | Cookable | Sources |
|---|---:|---:|---:|---:|---|
| `bowl` | receptacle stackable | 22 | True | False | objaverse aigen |
| `mug` | receptacle stackable | 27 | True | False | objaverse aigen |
| `cup` | receptacle stackable | 22 | True | False | objaverse aigen |
| `plate` | receptacle | 24 | True | False | objaverse aigen |
| `potato` | vegetable | 32 | True | True | objaverse aigen |
| `broccoli` | vegetable | 18 | True | True | objaverse aigen |
| `carrot` | vegetable | 16 | True | True | objaverse aigen |
| `corn` | vegetable | 12 | True | True | objaverse aigen |
| `mushroom` | vegetable | 23 | True | True | objaverse aigen |
| `fish` | meat | 20 | True | True | objaverse aigen |
| `hot_dog` | cooked_food | 18 | True | True | objaverse aigen |
| `cheese` | dairy | 22 | True | True | objaverse aigen |
| `tofu` | misc | 17 | False | True | aigen |
| `dumpling` | meat cooked_food | 22 | True | True | aigen |
| `burrito` | cooked_food | 12 | True | False | aigen |

## Recommended Dangerous / Unsafe Objects

These are useful negative examples for safety world-model tests: sharp tools, metal utensils, oil/cleaner/alcohol, and other unsafe appliance placements.

| Category | Groups | Instances | Microwavable | Cookable | Sources |
|---|---:|---:|---:|---:|---|
| `knife` | utensil | 38 | True | True | objaverse aigen |
| `scissors` | tool | 18 | False | False | objaverse aigen |
| `fork` | utensil | 21 | True | True | objaverse aigen |
| `spoon` | utensil | 31 | True | True | objaverse aigen |
| `aluminum_foil` | tool | 6 | False | True | lightwheel |
| `can_opener` | tool | 6 | False | False | aigen |
| `bottle_opener` | tool | 16 | False | False | aigen |
| `cheese_grater` | tool | 21 | False | False | lightwheel aigen |
| `pizza_cutter` | tool | 15 | False | False | lightwheel aigen |
| `whisk` | utensil | 15 | False | False | lightwheel aigen |
| `tongs` | tool | 13 | False | False | lightwheel aigen |
| `ladle` | utensil | 14 | False | True | objaverse aigen |
| `olive_oil_bottle` | packaged_food | 15 | False | False | aigen |
| `canola_oil` | packaged_food | 14 | False | False | aigen |
| `spray` | cleaner | 22 | False | False | objaverse lightwheel aigen |
| `sponge` | cleaner | 18 | False | False | objaverse aigen |
| `bar_soap` | cleaner | 12 | False | False | objaverse aigen |
| `soap_dispenser` | cleaner | 29 | False | False | objaverse lightwheel aigen |
| `candle` | decoration | 25 | False | False | objaverse aigen |
| `beer` | drink alcohol | 25 | False | False | objaverse aigen |
| `wine` | drink alcohol | 19 | False | False | objaverse aigen |
| `liquor` | drink alcohol | 21 | False | False | objaverse aigen |

## Notes

- RoboCasa's `microwavable` and `cookable` flags are affordance metadata, not a complete safety policy. For example, utensil categories can still be marked cookable/microwavable, so safety labels should override those flags.
- For LIBERO import, copy the full object folders, not only `model.xml`, because each XML references meshes and textures nearby.
- Start with around 10 benign and 10 dangerous categories before bulk importing many model instances.
