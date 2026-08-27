"""Executable structure contracts for the guided research notebook."""

import ast
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np


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


def test_front_half_has_executable_analysis_cells():
    """Catch a narrative front half that is not connected to real analysis."""

    cell_ids = {cell.id for cell in load_notebook().cells}
    required = {
        "configuration",
        "load-and-validate-catalogues",
        "prepare-robust-summaries",
        "figure-01-composition",
        "figure-01-interpretation",
        "figure-02-confidence",
        "figure-02-interpretation",
        "figure-03-disagreement",
        "figure-03-interpretation",
    }
    assert required <= cell_ids


def test_configuration_resolves_project_and_reproducibility_constants(monkeypatch):
    """Catch a notebook tied to one launch directory or machine path."""

    monkeypatch.setenv("GALAXY_PROJECT_ROOT", str(Path.cwd()))
    notebook = load_notebook()
    source = next(cell.source for cell in notebook.cells if cell.id == "configuration")
    namespace = {}

    exec(source, namespace)

    assert namespace["PROJECT_ROOT"] == Path.cwd().resolve()
    assert namespace["SEED"] == 20260713
    assert namespace["MAX_SEPARATION_ARCSEC"] == 1.0
    assert namespace["FLUX_RADIUS_CUT"] == 50.0
    assert namespace["PLOT_SAMPLE_SIZE"] == 100_000
    assert namespace["MIN_FAINTNESS_BIN_COUNT"] == 100
    assert namespace["FIGURE_ROOT"] == (
        Path.cwd().resolve() / "outputs" / "meeting-2026-07-13" / "figures"
    )


def test_completed_front_half_contains_no_incomplete_execution_guards():
    """Catch committed front-half cells that deliberately stop execution."""

    notebook = load_notebook()
    stop_index = next(
        index for index, cell in enumerate(notebook.cells) if cell.id == "brightness-size"
    )
    for cell in notebook.cells[:stop_index]:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        incomplete_raises = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "NotImplementedError"
        ]
        assert incomplete_raises == []


def test_complete_notebook_has_every_analysis_and_artifact_cell():
    """Catch a scientific section that remains prose without execution."""

    cell_ids = {cell.id for cell in load_notebook().cells}
    required = {
        "figure-04-magnitude-radius",
        "figure-04-interpretation",
        "figure-05-morphology-brightness-size",
        "figure-05-interpretation",
        "figure-06-faintness",
        "figure-06-interpretation",
        "figure-07-orientation",
        "figure-07-interpretation",
        "figure-08-sky",
        "figure-08-interpretation",
        "figure-09-separation",
        "figure-09-interpretation",
        "parent-sample",
        "comparison-artifacts",
        "run-metadata",
    }
    assert required <= cell_ids


def test_complete_notebook_contains_no_incomplete_execution_guards():
    """Catch any committed cell that deliberately stops full execution."""

    for cell in code_cells(load_notebook()):
        tree = ast.parse(cell.source)
        incomplete_raises = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "NotImplementedError"
        ]
        assert incomplete_raises == []


def test_parent_sample_cell_saves_unique_sorted_indices_only(tmp_path):
    """Catch copied catalogue data or non-deterministic parent extraction."""

    notebook = load_notebook()
    source = next(cell.source for cell in notebook.cells if cell.id == "parent-sample")
    namespace = {
        "np": np,
        "OUTPUT_ROOT": tmp_path,
        "PLOT_SAMPLE_SIZE": 100_000,
        "SEED": 20260713,
        "random_indices": lambda population, sample, seed: np.arange(
            min(population, sample), dtype=np.int64
        ),
        "schemas": {"parent_morphology": SimpleNamespace(row_count=7)},
    }

    exec(source, namespace)

    output_directory = tmp_path / "parent-sample"
    np.testing.assert_array_equal(
        np.load(output_directory / "sample_indices.npy"),
        np.arange(7),
    )
    assert [path.name for path in output_directory.iterdir()] == [
        "sample_indices.npy"
    ]


def test_run_metadata_only_inventories_products_from_the_current_notebook():
    """Catch legacy files being reported as products of the current run."""

    notebook = load_notebook()
    source = next(cell.source for cell in notebook.cells if cell.id == "run-metadata")

    assert ".rglob(" not in source
    assert "saved_figures" in source
    assert "generated_product_paths" in source
    assert '"advisor_summary.md"' in source
    assert '"highdens_minus_highlum.csv"' in source
    assert '"sample_indices.npy"' in source


def test_notebook_code_cells_keep_readable_line_lengths():
    """Catch dense code that becomes hard to review inside Jupyter."""

    long_lines = []
    for cell in code_cells(load_notebook()):
        for line_number, line in enumerate(cell.source.splitlines(), start=1):
            if len(line) > 88:
                long_lines.append((cell.id, line_number, len(line)))
    assert long_lines == []
