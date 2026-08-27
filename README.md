# Scientific Research at UFRJ: Galaxies and AI 🌌🤖

Welcome to the **`scientific-research-at-ufrj-galaxies-and-AI`** repository!

This repository contains an undergraduate scientific-research project at the
**Universidade Federal do Rio de Janeiro (UFRJ)**. The project studies galaxy
morphology in large astronomical catalogues and develops reproducible tools for
connecting astrophysical questions with machine-learning classifications.

The current study implemented in this repository compares two Dark Energy
Survey (DES) morphology cross-matches known locally as `highlum` and
`highdens`. It validates the catalogues, constructs robust early-type and
late-type galaxy samples, compares their measured distributions, saves
advisor-ready Matplotlib figures, and records the limits of every
interpretation.

---

## 📖 Project Overview

Modern surveys contain millions of sources, making manual morphological
classification impractical. Machine-learning catalogues can extend this work to
survey scale, but their outputs still require careful validation. Differences
between selected catalogues may reflect their construction, observational
effects, model behaviour, or genuine astrophysical structure.

The present workflow therefore follows a cautious order:

1. document the origin and role of each catalogue;
2. validate identifiers, coordinates, probabilities, flags, and cross-match
   separations;
3. define robust morphology samples from the published catalogue flags;
4. compare sample composition and observable distributions;
5. measure disagreement among the five morphology models;
6. save figures, tables, metadata, and an advisor-ready summary; and
7. distinguish measured associations from unsupported causal conclusions.

The primary analysis is
[`notebooks/01_robust_morphology_environment_analysis.ipynb`](notebooks/01_robust_morphology_environment_analysis.ipynb).
Its detailed methodological guide explains every input variable, derived
quantity, graph, result, and limitation in
[`notebooks/README.md`](notebooks/README.md).

### Current research scope

| Area | Repository status |
| --- | --- |
| DES morphology catalogue inspection | Implemented in `src/preview_catalog.py`. |
| Robust `highlum`/`highdens` comparison | Implemented in the primary notebook and `src/galaxy_analysis/`. |
| Reproducible figures and research tables | Implemented; generated products remain outside Git. |
| Morphometric analysis with Morfometryka variables | Deferred until the corrected catalogue is available. |
| Galaxy cluster finding with machine learning | Literature-study direction; not yet implemented here. |
| Causal environmental interpretation | Outside the current evidence and intentionally not claimed. |

---

## 🗂️ Repository Structure

The tree below represents the current tracked source structure. Local FITS
catalogues, article PDFs, generated figures, notebook outputs, and Python cache
files are intentionally excluded from Git.

```text
scientific-research-at-ufrj-galaxies-and-AI/
├── data/
│   └── README.md                         # Local catalogue layout and manifest
├── docs/
│   └── articles/
│       ├── README.md                     # Reading list and note-taking guidance
│       ├── notes/.gitkeep                # Tracked research-note directory
│       └── pdfs/.gitkeep                 # Ignored local literature PDFs
├── graphs/
│   └── .gitkeep                         # Ignored preview-script PNG outputs
├── notebooks/
│   ├── 01_robust_morphology_environment_analysis.ipynb
│   └── README.md                         # Full scientific and variable guide
├── src/
│   ├── galaxy_analysis/
│   │   ├── __init__.py
│   │   ├── artifacts.py                 # Paths and saved research artifacts
│   │   ├── catalog.py                   # Memory-conscious FITS access
│   │   ├── plotting.py                  # Reusable Matplotlib figures
│   │   ├── quality.py                   # Catalogue quality gates
│   │   ├── reporting.py                 # Natural-language interpretations
│   │   ├── selection.py                 # Robust masks and filter audits
│   │   └── statistics.py                # Descriptive statistical summaries
│   └── preview_catalog.py                # Standalone DES catalogue preview
├── tests/
│   ├── conftest.py                        # Shared synthetic FITS fixture
│   ├── test_artifacts.py
│   ├── test_catalog.py
│   ├── test_notebook_structure.py
│   ├── test_plotting.py
│   ├── test_quality.py
│   ├── test_reporting.py
│   ├── test_selection.py
│   └── test_statistics.py
├── .gitignore
├── requirements.txt
└── README.md
```

### Component responsibilities

| Path | Responsibility |
| --- | --- |
| `data/` | Holds the tracked manifest and the ignored local FITS hierarchy. |
| `docs/articles/` | Organizes the reading list, local PDFs, and future research notes. |
| `notebooks/` | Contains the narrated scientific analysis and its academic guide. |
| `src/galaxy_analysis/` | Provides small, tested modules used by the notebook. |
| `src/preview_catalog.py` | Previews the parent DES catalogue and saves exploratory graphs. |
| `tests/` | Tests catalogue access, selections, statistics, plots, reporting, artifacts, and notebook structure. |
| `graphs/` | Receives ignored PNG files from the standalone preview script. |
| `outputs/` | Created at runtime for ignored notebook figures, tables, metadata, and summaries. |

---

## 💾 Data Organization and Repository Hygiene

The astronomical catalogues are several megabytes to gigabytes in size and may
have redistribution restrictions. They belong in `data/` on the local machine,
but they do **not** belong in Git history.

The analysis expects this logical layout:

```text
data/
├── DES_DR1_CNN_morphological_catalog.fit
├── raw/
│   └── red-sequence/
│       ├── redspell_highdens7_final.fits
│       └── redspell_highlum7_final.fits
└── processed/
    └── crossmatches/
        └── vega-ferrero/
            ├── match_VF_highdens.fits
            └── match_VF_highlum.fits
```

See [`data/README.md`](data/README.md) for the catalogue manifest, row counts,
roles, and robust classification flags. The aliases `highlum` and `highdens`
identify upstream source selections; the current analysis does not treat those
names as direct physical measurements of luminosity or environmental density.

Repository hygiene rules:

- FITS catalogues remain local and ignored.
- Article PDFs remain local; reproducible reading notes may be committed.
- `graphs/*.png` and the entire `outputs/` tree are regenerated, not tracked.
- Notebook checkpoints, Python caches, isolated worktrees, meeting
  transcripts, and temporary specifications remain outside commits.
- The primary notebook is committed without execution outputs.

---

## 🧪 Analysis Design and Reproducibility

The notebook contains the scientific narrative, while reusable logic lives in
tested Python modules. This separation keeps the calculations reviewable without
turning the notebook into a long collection of opaque helper functions.

The implemented workflow includes:

- memory-mapped FITS inspection and selective column loading;
- explicit quality gates before scientific comparison;
- robust early-type (`FLAG_LTG == 4`) and late-type (`FLAG_LTG == 5`) masks;
- composition fractions with Wilson confidence intervals;
- descriptive statistics and fixed-percentile summaries;
- deterministic sampling for large visualizations;
- shared axes, bins, and color scales for fair catalogue comparisons;
- nine Matplotlib figures with stable filenames;
- CSV and JSON research artifacts plus natural-language interpretations; and
- tests that enforce clean notebook cells, readable code, and scientific
  interpretation boundaries.

Generated notebook products are written below
`outputs/meeting-2026-07-13/`. The complete artifact inventory and the meaning
of every graph are documented in [`notebooks/README.md`](notebooks/README.md).

---

## 🛠️ Tools and Technologies

The current environment is intentionally focused on the implemented analysis:

| Purpose | Tools |
| --- | --- |
| Astronomical FITS data | `Astropy` |
| Numerical analysis | `NumPy` |
| Visualization | `Matplotlib` |
| Interactive research | `JupyterLab`, `nbclient`, `nbconvert`, `nbformat` |
| Verification | `pytest` |

Exact minimum versions are listed in [`requirements.txt`](requirements.txt).

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mayraesilva/scientific-research-at-ufrj-galaxies-and-AI.git
cd scientific-research-at-ufrj-galaxies-and-AI
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv scientific-research
source scientific-research/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
scientific-research\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Place the local catalogues

Obtain the authorized data files and place them in the paths described in
[`data/README.md`](data/README.md) and the data tree above. Do not add the FITS
files to Git.

### 5. Run the primary notebook

```bash
jupyter lab notebooks/01_robust_morphology_environment_analysis.ipynb
```

Run the notebook cells from top to bottom. The notebook validates the expected
inputs before creating its ignored artifacts.

### 6. Preview the parent DES morphology catalogue

```bash
python src/preview_catalog.py --rows 50 --analysis-rows 100000
```

This command prints a small catalogue preview and saves exploratory PNG files
under `graphs/`. Use `python src/preview_catalog.py --help` to inspect optional
catalogue and output-path arguments.

### 7. Run the verification suite

```bash
MPLBACKEND=Agg python -m pytest -q
```

The tests use synthetic FITS data where possible, so the code and notebook
structure can be checked without loading the large research catalogues.

---

## 📚 Foundational Literature and References

The current work is informed by:

- Cheng, T.-Y., et al. (2021). *Galaxy Morphological Classification Catalogue
  of the Dark Energy Survey Year 3 Data with Convolutional Neural Networks*.
- Ferrari, F., de Carvalho, R. R., & Trevisan, M. (2015). *Morfometryka—A New
  Way of Establishing Morphological Classification of Galaxies*. The
  Astrophysical Journal, 814, 55.
- Karttunen, H., et al. *Fundamental Astronomy*, 5th edition.
- Tian, D.-C., et al. (2025). *COSMIC: A Galaxy Cluster-Finding Algorithm Using
  Machine Learning*.
- Vega-Ferrero, J., et al. (2021). *Pushing Automated Morphological
  Classifications to Their Limits with the Dark Energy Survey*.

The local reading list and note-taking guidance are maintained in
[`docs/articles/README.md`](docs/articles/README.md).

---

## 📍 About

This repository is maintained by an undergraduate Physics student pursuing a
bachelor's degree at the **Universidade Federal do Rio de Janeiro (UFRJ)** and
conducting scientific research that connects astrophysics with modern
computational methods.

**Research Advisor:** Professor Arianna Cortesi

### Contact

- **Student:** [mayraeduarda2002@gmail.com](mailto:mayraeduarda2002@gmail.com)
- **Research Advisor:** [aricorte@gmail.com](mailto:aricorte@gmail.com)

*Viva a ciência brasileira!* 🇧🇷🔬
