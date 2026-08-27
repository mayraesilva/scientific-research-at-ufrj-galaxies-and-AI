"""Executable structure contracts for the guided research notebook."""

import ast
from pathlib import Path

import nbformat


NOTEBOOK = Path("notebooks/01_robust_morphology_environment_analysis.ipynb")

EXPECTED_SECTION_IDS = [
    "opening",
    "research-questions",
    "catalogue-language",
    "reproducible-setup",
    "quality-gate",
    "composition",
    "classification-confidence",
    "brightness-size",
    "faintness",
    "orientation",
    "sky-footprint",
    "match-quality",
    "conclusions",
]

PRIMARY_FIGURES = [
    "01_catalogue_composition.png",
    "02_robust_ltg_probability.png",
    "03_model_disagreement.png",
    "04_magnitude_half_light_radius.png",
    "05_morphology_brightness_size.png",
    "06_probability_vs_faintness.png",
    "07_ltg_vs_edgeon_probability.png",
    "08_sky_footprint.png",
    "09_match_separation.png",
]


def load_notebook():
    """Load the tracked notebook without executing it."""

    return nbformat.read(NOTEBOOK, as_version=4)


def code_cells(notebook):
    """Return code cells so output and syntax contracts share one definition."""

    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def test_notebook_is_valid_version_four_document():
    notebook = load_notebook()
    nbformat.validate(notebook)
    assert notebook.nbformat == 4


def test_every_code_cell_contains_valid_python_syntax():
    for cell in code_cells(load_notebook()):
        ast.parse(cell.source, filename=f"notebook-cell-{cell.id}")


def test_notebook_is_committed_without_outputs():
    cells = code_cells(load_notebook())
    assert cells
    assert all(cell.execution_count is None for cell in cells)
    assert all(cell.outputs == [] for cell in cells)


def test_notebook_uses_project_kernel_metadata():
    notebook = load_notebook()
    assert notebook.metadata.kernelspec.name == "python3"
    assert notebook.metadata.language_info.name == "python"


def test_notebook_follows_the_approved_research_story():
    ordered_ids = [cell.id for cell in load_notebook().cells]
    positions = [ordered_ids.index(cell_id) for cell_id in EXPECTED_SECTION_IDS]
    assert positions == sorted(positions)


def test_each_scientific_figure_has_stable_semantic_metadata():
    figure_cells = [
        cell
        for cell in load_notebook().cells
        if "scientific-figure" in cell.metadata.get("tags", [])
    ]
    filenames = [cell.metadata["figure_filename"] for cell in figure_cells]
    assert filenames == PRIMARY_FIGURES


def test_each_figure_is_followed_by_result_and_limitation_cells():
    notebook = load_notebook()
    for index, cell in enumerate(notebook.cells):
        if "scientific-figure" not in cell.metadata.get("tags", []):
            continue

        following_tags = [
            tag
            for following in notebook.cells[index + 1 : index + 3]
            for tag in following.metadata.get("tags", [])
        ]
        assert "result-interpretation" in following_tags
        assert "scientific-limitation" in following_tags


def test_notebook_keeps_reusable_functions_in_tested_modules():
    for cell in code_cells(load_notebook()):
        tree = ast.parse(cell.source)
        definitions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert definitions == []


def test_every_code_cell_opens_with_an_intent_comment():
    for cell in code_cells(load_notebook()):
        first_nonempty_line = next(
            line.strip() for line in cell.source.splitlines() if line.strip()
        )
        assert first_nonempty_line.startswith("#")


def test_notebook_plotting_imports_are_matplotlib_only():
    imported_roots = set()
    for cell in code_cells(load_notebook()):
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])

    assert "matplotlib" in imported_roots
    assert "seaborn" not in imported_roots
