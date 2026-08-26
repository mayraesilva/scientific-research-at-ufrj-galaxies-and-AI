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
