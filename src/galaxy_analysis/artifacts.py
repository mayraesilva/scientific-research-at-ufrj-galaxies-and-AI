"""Write reproducible notebook artifacts outside version-controlled inputs.

The scientific calculation and plotting modules remain free of file writes.
This module owns the small amount of orchestration needed to locate the
project, save a Matplotlib figure, and serialize run metadata consistently.
"""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
import numpy as np


def _json_default(value: Any) -> Any:
    """Convert common research scalar, array, and path types for JSON."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def find_project_root(start: Path, configured: Path | None = None) -> Path:
    """Locate the repository without embedding a machine-specific path.

    An explicit configured location takes precedence.  Otherwise, the search
    begins at ``start`` and walks upward, which supports Jupyter sessions
    launched from the repository root, the notebooks directory, or a worktree.
    """

    candidates = [configured] if configured else [start, *start.parents]

    for candidate in candidates:
        resolved = Path(candidate).resolve()
        has_requirements = (resolved / "requirements.txt").is_file()
        has_data_manifest = (resolved / "data" / "README.md").is_file()
        if has_requirements and has_data_manifest:
            return resolved

    raise FileNotFoundError(
        "project root requires requirements.txt and data/README.md"
    )


def save_figure(
    figure: Figure,
    directory: Path,
    filename: str,
    dpi: int = 150,
) -> Path:
    """Save one PNG below the configured figure directory and return its path.

    Requiring a plain filename prevents a plot call from quietly escaping the
    ignored output directory.  A stable name means rerunning the notebook
    replaces the previous artifact instead of accumulating ambiguous copies.
    """

    if Path(filename).name != filename or not filename.endswith(".png"):
        raise ValueError("filename must be a plain .png name")

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write deterministic, readable JSON and return its artifact path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return path
