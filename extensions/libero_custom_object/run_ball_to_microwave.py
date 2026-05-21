import sys
from pathlib import Path

from run_pick_place import main


EXTENSION_ROOT = Path(__file__).resolve().parent
DEFAULT_BDDL = (
    EXTENSION_ROOT
    / "libero_custom_object"
    / "bddl_files"
    / "check_open_microwave_ball_on_table.bddl"
)
DEFAULT_OUTPUT = EXTENSION_ROOT / "outputs" / "ball_to_microwave.mp4"


def main_with_defaults():
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        args.insert(0, "ball")

    defaults = [
        ("--bddl", str(DEFAULT_BDDL)),
        ("--target-site", "microwave_1_heating_region"),
        ("--placement-mode", "center"),
        ("--approach-from-site-axis", "local_neg_y"),
        ("--target-approach-distance", "0.18"),
        ("--transport-z", "1.14"),
        ("--place-clearance", "0.0"),
        ("--camera", "frontview"),
        ("--output", str(DEFAULT_OUTPUT)),
    ]
    for option, value in defaults:
        if option not in args:
            args.extend([option, value])

    sys.argv = [sys.argv[0], *args]
    main()


if __name__ == "__main__":
    main_with_defaults()
