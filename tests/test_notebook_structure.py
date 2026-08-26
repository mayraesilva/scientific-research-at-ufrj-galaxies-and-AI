from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np


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
    assert namespace["COMPARISON_MAGNITUDE_LIMITS"] == (21.6, 18.0)
    assert namespace["COMPARISON_RADIUS_LIMITS"] == (2.5, 22.0)
    assert namespace["SUMMARY_HEADERS"] == (
        "catalog", "class", "variable", "n_total", "n_valid", "n_missing",
        "mean", "median", "std_ddof1", "min", "p05", "p25", "p50", "p75",
        "p95", "max", "iqr",
    )
    assert namespace["FILTER_AUDIT_HEADERS"] == (
        "catalog", "stage", "rule", "n_before", "n_removed", "n_after",
        "fraction_removed",
    )


def test_parent_sample_cell_requests_reproducible_indices_only(tmp_path):
    notebook = load_notebook()
    source = next(cell.source for cell in notebook.cells if cell.id == "parent-sample-code")
    calls = []

    def fake_random_indices(population_size, sample_size, seed):
        calls.append((population_size, sample_size, seed))
        return np.arange(min(population_size, sample_size), dtype=np.int64)

    namespace = {
        "np": np,
        "OUTPUT_ROOT": tmp_path,
        "PROJECT_ROOT": tmp_path,
        "PLOT_SAMPLE_SIZE": 100_000,
        "SEED": 20260713,
        "random_indices": fake_random_indices,
        "schemas": {"parent_morphology": SimpleNamespace(row_count=7)},
    }
    exec(source, namespace)
    assert calls == [(7, 100_000, 20260713)]
    np.testing.assert_array_equal(
        np.load(tmp_path / "parent-sample" / "sample_indices.npy"), np.arange(7)
    )
    assert [path.name for path in (tmp_path / "parent-sample").iterdir()] == [
        "sample_indices.npy"
    ]


def test_notebook_saves_every_required_graph_at_inspection_resolution():
    notebook = load_notebook()
    all_code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    per_catalog = {
        "magnitude_vs_flux_radius_all.png",
        "magnitude_vs_flux_radius_cut50.png",
        "magnitude_vs_flux_radius_density.png",
        "magnitude_size_robust_classes.png",
        "mp_ltg_by_robust_class.png",
        "mp_edgeon_by_robust_class.png",
        "model_dispersion_ltg.png",
        "ltg_probability_vs_magnitude.png",
        "edgeon_vs_ltg_probability.png",
        "sky_distribution_all.png",
        "sky_distribution_robust_classes.png",
        "flag_ltg_counts.png",
    }
    comparison = {
        "robust_class_fractions_comparison.png",
        "morphology_parameter_comparison.png",
        "separation_distribution_comparison.png",
    }
    assert "figure.savefig(path, dpi=150, bbox_inches='tight')" in all_code
    assert all(all_code.count(filename) == 2 for filename in per_catalog)
    assert all(all_code.count(filename) == 1 for filename in comparison)


def test_robust_magnitude_radius_figures_use_identical_comparison_limits():
    notebook = load_notebook()
    for cell_id in ("magnitude-radius-classes-plot", "highdens-figures"):
        source = next(cell.source for cell in notebook.cells if cell.id == cell_id)
        assert "set_xlim(COMPARISON_MAGNITUDE_LIMITS)" in source
        assert "set_ylim(COMPARISON_RADIUS_LIMITS)" in source
