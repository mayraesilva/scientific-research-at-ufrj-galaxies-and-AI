from pathlib import Path

import nbformat


NOTEBOOK = Path("notebooks/01_robust_morphology_environment_analysis.ipynb")


def load_notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_notebook_is_valid_version_four_document():
    notebook = load_notebook()
    nbformat.validate(notebook)
    assert notebook.nbformat == 4


def test_notebook_is_committed_without_outputs():
    notebook = load_notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(cell.outputs == [] for cell in code_cells)


def test_notebook_uses_project_kernel_metadata():
    notebook = load_notebook()
    assert notebook.metadata.kernelspec.name == "python3"
    assert notebook.metadata.language_info.name == "python"


def test_configuration_cell_resolves_project_and_reproducibility_constants(monkeypatch):
    monkeypatch.setenv("GALAXY_PROJECT_ROOT", str(Path.cwd()))
    notebook = load_notebook()
    source = next(cell.source for cell in notebook.cells if cell.id == "configuration")
    namespace = {}
    exec(source, namespace)
    assert namespace["np"].__name__ == "numpy"
    assert namespace["PROJECT_ROOT"] == Path.cwd().resolve()
    assert namespace["SEED"] == 20260713
    assert namespace["MAX_SEPARATION_ARCSEC"] == 1.0
    assert namespace["FLUX_RADIUS_CUT"] == 50.0
    assert namespace["PLOT_SAMPLE_SIZE"] == 100_000
    assert namespace["SUMMARY_HEADERS"] == (
        "catalog", "class", "variable", "n_total", "n_valid", "n_missing",
        "mean", "median", "std_ddof1", "min", "p05", "p25", "p50", "p75",
        "p95", "max", "iqr",
    )
    assert namespace["FILTER_AUDIT_HEADERS"] == (
        "catalog", "stage", "rule", "n_before", "n_removed", "n_after",
        "fraction_removed",
    )
