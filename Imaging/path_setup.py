from pathlib import Path
import sys


def ensure_project_paths(script_path):
    """Make the Imaging project layout importable regardless of cwd."""
    script_path = Path(script_path).resolve()

    imaging_root = None
    for parent in (script_path.parent, *script_path.parents):
        if parent.name == "Imaging":
            imaging_root = parent
            break

    if imaging_root is None:
        imaging_root = script_path.parent

    candidate_paths = [
        imaging_root,
        imaging_root / "01_lidc-preprocessing",
        imaging_root / "02_se-resnet50",
        imaging_root / "03_CIR",
        imaging_root / "04_gradnorm",
        imaging_root / "05_gradcam",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    return imaging_root
