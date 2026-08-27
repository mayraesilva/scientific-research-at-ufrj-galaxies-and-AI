"""Behavior tests for local, reproducible notebook artifacts."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image
import pytest

from src.galaxy_analysis.artifacts import (
    find_project_root,
    save_figure,
    write_json,
)


def test_find_project_root_walks_up_from_notebooks(tmp_path):
    """Catch root discovery that only works from one launch directory."""

    root = tmp_path / "project"
    nested = root / "notebooks"
    nested.mkdir(parents=True)
    (root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "README.md").write_text("data\n", encoding="utf-8")

    assert find_project_root(nested) == root.resolve()


def test_save_figure_creates_a_readable_stable_png(tmp_path):
    """Catch missing, empty, or unreadable graph artifacts."""

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    path = save_figure(figure, tmp_path / "figures", "result.png")

    assert path == tmp_path / "figures" / "result.png"
    assert path.stat().st_size > 0
    with Image.open(path) as image:
        assert image.size[0] > 0
        assert image.size[1] > 0
    plt.close(figure)


def test_save_figure_rejects_paths_disguised_as_filenames(tmp_path):
    """Catch accidental writes outside the configured figure directory."""

    figure, _ = plt.subplots()
    with pytest.raises(ValueError, match="plain .png name"):
        save_figure(figure, tmp_path, "nested/result.png")
    plt.close(figure)


def test_write_json_round_trips_the_same_payload(tmp_path):
    """Catch metadata that cannot be read back as the supplied values."""

    payload = {"seed": 20260713, "files": ["one.png"]}

    path = write_json(tmp_path / "metadata.json", payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
