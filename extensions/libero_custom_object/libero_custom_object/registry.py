import pathlib
import re

import numpy as np
import yaml
from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import OBJECTS_DICT, register_object


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
ASSETS_ROOT = PACKAGE_ROOT / "assets"
MANIFEST_PATH = ASSETS_ROOT / "manifest.yaml"
BDDL_DIR = PACKAGE_ROOT / "bddl_files"
SAMPLE_BDDL_NAME = "pick_the_custom_alphabet_soup_and_place_it_in_the_basket.bddl"
TWO_OBJECT_BDDL_NAME = (
    "pick_the_custom_alphabet_soup_and_ketchup_and_place_them_in_the_basket.bddl"
)
KNIFE_BDDL_NAME = "pick_the_custom_knife_and_place_it_in_the_basket.bddl"
SCISSORS_BDDL_NAME = "pick_the_custom_scissors_and_place_them_in_the_basket.bddl"
HAMMER_BDDL_NAME = "pick_the_custom_hammer_and_place_it_in_the_basket.bddl"
HAZARD_TOOLS_BDDL_NAME = (
    "pick_the_custom_scissors_and_hammer_and_place_them_in_the_basket.bddl"
)
STEAK_KNIFE_BDDL_NAME = "pick_the_custom_steak_knife_and_place_it_in_the_basket.bddl"
CAN_OPENER_BDDL_NAME = "pick_the_custom_can_opener_and_place_it_in_the_basket.bddl"
KITCHEN_HAZARDS_BDDL_NAME = (
    "pick_the_custom_steak_knife_and_can_opener_and_place_them_in_the_basket.bddl"
)
FOUR_OBJECTS_BDDL_NAME = (
    "pick_the_custom_knife_scissors_hammer_and_alphabet_soup_and_place_them_in_the_basket.bddl"
)
KITCHEN_MICROWAVE_OPEN_BDDL_NAME = "check_kitchen_microwave_open_on_table.bddl"
KITCHEN_STOVE_BDDL_NAME = "check_kitchen_stove_on_table.bddl"
GIFT_BOX_HAZARDS_BDDL_NAME = "check_gift_box_turbosquid_hazards.bddl"
_REGISTERED_CATEGORIES = set()


class CustomExtensionObject(MujocoXMLObject):
    def __init__(self, name, xml_path, category_name, rotation, rotation_axis):
        super().__init__(
            str(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = category_name
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.object_properties = {"vis_site_names": {}}


def get_manifest_path():
    return str(MANIFEST_PATH)


def get_sample_bddl_path():
    return str(BDDL_DIR / SAMPLE_BDDL_NAME)


def get_two_object_bddl_path():
    return str(BDDL_DIR / TWO_OBJECT_BDDL_NAME)


def get_knife_bddl_path():
    return str(BDDL_DIR / KNIFE_BDDL_NAME)


def get_scissors_bddl_path():
    return str(BDDL_DIR / SCISSORS_BDDL_NAME)


def get_hammer_bddl_path():
    return str(BDDL_DIR / HAMMER_BDDL_NAME)


def get_hazard_tools_bddl_path():
    return str(BDDL_DIR / HAZARD_TOOLS_BDDL_NAME)


def get_steak_knife_bddl_path():
    return str(BDDL_DIR / STEAK_KNIFE_BDDL_NAME)


def get_can_opener_bddl_path():
    return str(BDDL_DIR / CAN_OPENER_BDDL_NAME)


def get_kitchen_hazards_bddl_path():
    return str(BDDL_DIR / KITCHEN_HAZARDS_BDDL_NAME)


def get_four_objects_bddl_path():
    return str(BDDL_DIR / FOUR_OBJECTS_BDDL_NAME)


def get_kitchen_microwave_open_bddl_path():
    return str(BDDL_DIR / KITCHEN_MICROWAVE_OPEN_BDDL_NAME)


def get_kitchen_stove_bddl_path():
    return str(BDDL_DIR / KITCHEN_STOVE_BDDL_NAME)


def get_gift_box_hazards_bddl_path():
    return str(BDDL_DIR / GIFT_BOX_HAZARDS_BDDL_NAME)


def _class_name_from_category(category_name):
    return "".join(part.capitalize() for part in category_name.split("_"))


def _load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _validate_category_name(category_name, manifest_path):
    if not re.match(r"^[a-z][a-z0-9_]*$", category_name):
        raise ValueError(
            f"Invalid custom object category '{category_name}' in {manifest_path}"
        )


def register_custom_objects(manifest_path=None, assets_root=None):
    """Register manifest-backed custom objects into LIBERO's object registry."""
    manifest_path = pathlib.Path(manifest_path or MANIFEST_PATH).resolve()
    assets_root = pathlib.Path(assets_root or manifest_path.parent).resolve()
    manifest = _load_manifest(manifest_path)

    for entry in manifest.get("objects", []):
        category_name = entry["category_name"]
        asset_name = entry["asset_name"]
        xml_name = entry["xml"]
        rotation = tuple(entry.get("rotation", [np.pi / 2, np.pi / 2]))
        rotation_axis = entry.get("rotation_axis", "x")

        _validate_category_name(category_name, manifest_path)

        if category_name in _REGISTERED_CATEGORIES:
            continue
        if category_name in OBJECTS_DICT:
            raise ValueError(
                f"Cannot register custom object '{category_name}': category already exists in LIBERO."
            )

        xml_path = assets_root / "objects" / asset_name / xml_name
        if not xml_path.exists():
            raise FileNotFoundError(
                f"Custom object XML for '{category_name}' not found: {xml_path}"
            )

        def __init__(
            self,
            name=category_name,
            _xml_path=xml_path,
            _category_name=category_name,
            _rotation=rotation,
            _rotation_axis=rotation_axis,
        ):
            CustomExtensionObject.__init__(
                self,
                name=name,
                xml_path=_xml_path,
                category_name=_category_name,
                rotation=_rotation,
                rotation_axis=_rotation_axis,
            )

        cls = type(
            _class_name_from_category(category_name),
            (CustomExtensionObject,),
            {"__init__": __init__, "__module__": __name__},
        )
        register_object(cls)
        OBJECTS_DICT[category_name] = cls
        globals()[cls.__name__] = cls
        _REGISTERED_CATEGORIES.add(category_name)

    return sorted(_REGISTERED_CATEGORIES)
