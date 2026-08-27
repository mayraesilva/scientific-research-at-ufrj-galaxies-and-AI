# Galaxy Morphology with DES and Artificial Intelligence 🌌🤖

Welcome to the methodological and scientific guide for the robust morphology
analysis notebook!

This document explains the research motivation, catalogue terminology,
variables, analysis choices, figures, results, and limitations in the same
accessible astronomy-and-AI style used by the source repository. The scientific
language remains deliberately cautious: reproducible catalogue differences are
reported as associations, not as causal environmental effects.

**Companion notebook:**
[`01_robust_morphology_environment_analysis.ipynb`](01_robust_morphology_environment_analysis.ipynb)

---

## 📖 Project Overview

This notebook develops a reproducible descriptive comparison of two selected
galaxy catalogues, identified in the project as `highlum` and `highdens`, after
they were cross-matched with a Dark Energy Survey (DES) convolutional-neural-
network morphology catalogue. The analysis was motivated by a research meeting
held on 13 July 2026 and by the need to replace a collection of disconnected
exploratory plots with a scientifically ordered, auditable workflow.

The notebook first establishes whether the matched catalogues satisfy the
required schema, value-domain, identifier, and angular-separation conditions.
It then compares robust early-type-galaxy (ETG) and late-type-galaxy (LTG)
classifications, classification probabilities, disagreement among five neural
networks, apparent magnitude, image-plane half-light radius, viewing-
orientation output, sky coverage, and coordinate-match quality. Nine primary
Matplotlib figures organize this comparison. Numerical statements are derived
from the same calculated quantities shown in the figures, and every scientific
result is followed by an explicit limitation.

The analysis finds substantial differences between the two catalogue products,
including their robust-class fractions and probability distributions. These
differences are catalogue associations. The present data and documentation do
not support a causal claim that galaxy environment produced them, because the
physical source-selection functions represented by `highlum` and `highdens`
have not yet been fully documented.

**Keywords:** galaxy morphology; Dark Energy Survey; convolutional neural
networks; early-type galaxies; late-type galaxies; catalogue validation;
reproducible research.

---

## 🌌 Scientific Motivation

Modern imaging surveys contain far more galaxies than can be classified
manually. Automated morphology catalogues therefore use machine-learning models
to estimate whether an observed galaxy resembles an early-type system or a
late-type system and, separately, whether it is viewed edge-on. These outputs
make population-scale analysis possible, but they also introduce interpretive
risks. A model probability is not an independent truth label, agreement among
several related networks is not a complete uncertainty estimate, and an
apparent catalogue difference need not be a physical environmental effect.

The immediate research task was to inspect two locally supplied cross-match
products. Earlier exploratory code produced many graphs, but it did not always
state which scientific question each graph answered, why a particular sample
was selected, or what limitation prevented a stronger conclusion. The notebook
was therefore redesigned around a question-first narrative:

1. Are the supplied cross-match products internally suitable for comparison?
2. How many robust ETG and LTG classifications occur in each product?
3. How do the model outputs and observed image quantities differ?
4. Which conclusions are supported, and which decisions still require the
   advisor or additional data?

This order matters. Validation precedes interpretation, sample composition
precedes distribution comparison, and descriptive results precede physical
claims.

---

## 🗂️ Data and Catalogue Names

### Catalogue roles

| Notebook alias | Local file | Rows | Role in this study |
| --- | --- | ---: | --- |
| `parent_morphology` | `data/DES_DR1_CNN_morphological_catalog.fit` | 26,971,945 | Parent DES CNN morphology catalogue; only deterministic row indices are sampled in this notebook. |
| `highlum_source` | `data/raw/red-sequence/redspell_highlum7_final.fits` | 713,310 | Red-sequence source catalogue associated with the project label “high luminosity”; retained for provenance and not analyzed directly here. |
| `highdens_source` | `data/raw/red-sequence/redspell_highdens7_final.fits` | 1,482,074 | Red-sequence source catalogue associated with the project label “high density”; retained for provenance and not analyzed directly here. |
| `highlum` | `data/processed/crossmatches/vega-ferrero/match_VF_highlum.fits` | 34,768 | `highlum_source` records matched to the Vega-Ferrero morphology information; directly analyzed. |
| `highdens` | `data/processed/crossmatches/vega-ferrero/match_VF_highdens.fits` | 905,291 | `highdens_source` records matched to the Vega-Ferrero morphology information; directly analyzed. |

### What `highlum` means

`highlum` is a short project alias for the matched product derived from
`redspell_highlum7_final.fits`. The name indicates that the upstream source
selection was described as a **high-luminosity selection**. It does not, by
itself, define an absolute-luminosity threshold, a redshift interval, a cluster
membership rule, or an environmental category. Those physical selection
details are not encoded in the notebook or the current local data manifest.

Consequently, this README does not infer that every `highlum` row is
intrinsically more luminous than every `highdens` row. `MAG_AUTO_R` is an
apparent magnitude, not an absolute luminosity, and the physical meaning of the
upstream selection must be confirmed with the advisor or the catalogue-
construction documentation.

### What `highdens` means

`highdens` is the corresponding alias for the matched product derived from
`redspell_highdens7_final.fits`. Its name indicates that the upstream source
selection was described as a **high-density selection**. The notebook does not
contain the density estimator, aperture, neighbor definition, redshift window,
or threshold used to construct that selection.

The analysis therefore does not equate `highdens` with a measured continuous
environmental-density variable. It compares the supplied product with
`highlum` and reports the association. A causal environmental interpretation
is deliberately deferred.

### Why the source and matched catalogues are kept separate

The source catalogues describe the upstream red-sequence selections. The
matched products combine those selections with morphology fields. Only the
matched products contain the complete set of variables required for the nine
figures. Keeping the roles explicit prevents the analysis from silently
treating a parent catalogue, a source-selection catalogue, and a cross-match as
equivalent datasets.

---

## 📚 Input-Variable Dictionary

The notebook loads 21 columns from each matched product. The definitions below
describe how each name is used in this analysis.

### Identifiers and matched coordinates

| Variable | Meaning | Unit or domain | Use |
| --- | --- | --- | --- |
| `COADD_OBJECT_ID` | Identifier of the object in the DES morphology-side catalogue. | Integer identifier. | Checked for duplicates so one morphology object is not counted repeatedly. |
| `object_id` | Identifier inherited from the upstream red-sequence/source catalogue. | Integer identifier. | Checked independently for duplicates. |
| `RA_2` | Right ascension associated with the second table in the cross-match, interpreted as the morphology-side sky coordinate. | Degrees are assumed; the FITS column does not declare a unit. Valid range: `0 <= RA_2 < 360`. | Used to validate coordinates and display the sky footprint. |
| `DEC_2` | Declination associated with the second table in the cross-match. | Degrees are assumed; the FITS column does not declare a unit. Valid range: `-90 <= DEC_2 <= 90`. | Used to validate coordinates and display the sky footprint. |
| `Separation` | Angular distance between the two positions associated by the cross-match. | Arcseconds. Valid analysis range: `0 <= Separation <= 1`. | Used as a match-quality diagnostic, not as a physical galaxy distance. |

The suffix `_2` is a cross-match naming convention. It identifies columns
originating from the second matched table; it does not mean a second
observation or a second physical coordinate system.

### Observed photometric and image-plane quantities

| Variable | Meaning | Unit | Interpretive boundary |
| --- | --- | --- | --- |
| `MAG_AUTO_R` | DES r-band apparent magnitude measured with an elliptical Kron-like aperture. | Magnitudes. Smaller values are brighter. | It combines intrinsic luminosity, distance, attenuation, and measurement effects. It is not absolute luminosity. |
| `FLUX_RADIUS_R` | Radius of the circle containing half of the measured r-band object flux. | Image pixels. | It is an image-plane half-light radius, not a physical radius in parsecs or kiloparsecs. |

The `_R` suffix denotes the DES r band. The notebook inverts the magnitude axis
in brightness-size figures so that brighter objects appear on the left, in
accordance with the astronomical magnitude convention.

### Five-model LTG probabilities

| Variable | Meaning | Domain |
| --- | --- | --- |
| `P1_LTG` | Probability-like LTG output from model 1. | `[0, 1]` |
| `P2_LTG` | Probability-like LTG output from model 2. | `[0, 1]` |
| `P3_LTG` | Probability-like LTG output from model 3. | `[0, 1]` |
| `P4_LTG` | Probability-like LTG output from model 4. | `[0, 1]` |
| `P5_LTG` | Probability-like LTG output from model 5. | `[0, 1]` |
| `MP_LTG` | Median of `P1_LTG` through `P5_LTG`. | `[0, 1]` |

`LTG` means **late-type galaxy**. A larger LTG output indicates stronger model
support for the late-type side of the ETG/LTG classification. The five values
come from related neural-network models. They are not five independent physical
measurements, and their median must not be described as independently verified
classification accuracy.

`MP` means **median probability** in this catalogue. The notebook uses
`MP_LTG` as a continuous model summary, while `FLAG_LTG` supplies the catalogue
decision tier used for robust sample selection.

### Five-model edge-on probabilities

| Variable | Meaning | Domain |
| --- | --- | --- |
| `P1_EdgeOn` | Edge-on orientation output from model 1. | `[0, 1]` |
| `P2_EdgeOn` | Edge-on orientation output from model 2. | `[0, 1]` |
| `P3_EdgeOn` | Edge-on orientation output from model 3. | `[0, 1]` |
| `P4_EdgeOn` | Edge-on orientation output from model 4. | `[0, 1]` |
| `P5_EdgeOn` | Edge-on orientation output from model 5. | `[0, 1]` |
| `MP_EdgeOn` | Median of `P1_EdgeOn` through `P5_EdgeOn`. | `[0, 1]` |

The edge-on variables describe a predicted **viewing orientation**. They do not
form a third morphology class alongside ETG and LTG. Projection can obscure
disks and spiral structure, which is why the notebook compares `MP_EdgeOn` and
`MP_LTG` jointly rather than interpreting them as interchangeable labels.

### Catalogue flags

| Variable or value | Meaning in this notebook | Primary-sample status |
| --- | --- | --- |
| `FLAG_LTG` | Categorical ETG/LTG decision and confidence tier derived from agreement among the five LTG models. | Used for robust morphology selection. |
| `FLAG_LTG == 4` | Robust ETG classification, following Vega-Ferrero et al. (2021), Table 6. | Included as `robust_etg`. |
| `FLAG_LTG == 5` | Robust LTG classification, following Vega-Ferrero et al. (2021), Table 6. | Included as `robust_ltg`. |
| `FLAG_LTG` in `0, 1, 2, 3` | Lower-confidence/non-robust catalogue outcomes. Even values are ETG-side and odd values are LTG-side in the source encoding, but their finer tier distinctions are not used here. | Displayed together as `other_flags`; excluded from the primary robust comparison. |
| `FLAG_EdgeOn` | Categorical confidence/decision field associated with the edge-on models. | Loaded for catalogue completeness but not used to define the primary ETG/LTG sample. |

The primary masks use exact equality with flags 4 and 5. An even/odd shortcut
would incorrectly promote non-robust rows into the robust analysis and is
therefore prohibited.

---

## 🧮 Derived Analysis Variables

### Morphology masks and catalogue groups

| Derived name | Definition | Purpose |
| --- | --- | --- |
| `robust_etg` | `FLAG_LTG == 4` | Select robust early-type classifications. |
| `robust_ltg` | `FLAG_LTG == 5` | Select robust late-type classifications. |
| `robust_any` | `robust_etg OR robust_ltg` | Select every row admitted to the primary robust comparison. |
| `non_robust` | Logical complement of `robust_any`. | Keep excluded rows explicit. |
| `other_flags` | Composition-table group containing `FLAG_LTG` values 0–3. | Show how much of each matched catalogue is outside the primary robust sample. |
| `all_valid` | All finite rows for the variable being summarized. | Describe the complete matched product without applying the robust-class mask. |

### Model-disagreement variables

For each row, the five LTG outputs are assembled into a vector. The helper
`model_dispersion` calculates:

| Derived name | Definition | Interpretation |
| --- | --- | --- |
| `mean` | Arithmetic mean of `P1_LTG`–`P5_LTG`. | Alternate center of the five model outputs. |
| `median` | Median of `P1_LTG`–`P5_LTG`. | Equivalent conceptual quantity to `MP_LTG`. |
| `std_ddof1` | Sample standard deviation of the five outputs, using `ddof=1`. | Inter-model disagreement diagnostic. |
| `range` | Maximum minus minimum of the five outputs. | Full observed spread among the five models. |
| `P1_P5_LTG_STD` | Name used in the summary table for `std_ddof1`. | Quantity shown in Figure 3. |

These quantities do not include training-label uncertainty, common model bias,
image artifacts, source-selection effects, or domain shift. They are not a
complete predictive-uncertainty model.

### Binned faintness summaries

The combined finite `MAG_AUTO_R` range is divided into 12 fixed intervals. For
each catalogue and interval, the notebook calculates:

| Name | Meaning |
| --- | --- |
| `bin_edges` | Boundaries of the magnitude intervals. |
| `bin_centers` | Midpoint of each interval, used for plotting. |
| `count` | Number of finite `(MAG_AUTO_R, MP_LTG)` pairs in the interval. |
| `median` | Median `MP_LTG` in the interval. |
| `p25` | 25th percentile of `MP_LTG`. |
| `p75` | 75th percentile of `MP_LTG`. |

Only bins with at least `MIN_FAINTNESS_BIN_COUNT = 100` rows enter the orange
median line, interquartile band, and natural-language endpoint comparison. This
rule prevents a one-object bright bin from being presented as a catalogue-wide
trend.

### Descriptive-statistics fields

Each `SummaryRow` stored in `summary_statistics.csv` contains:

| Field | Meaning |
| --- | --- |
| `catalog` | Catalogue alias: `highlum` or `highdens`. |
| `class_name` | `all_valid`, `robust_etg`, or `robust_ltg`. |
| `variable` | Scientific variable summarized. |
| `n_total` | Number of rows supplied to the summary. |
| `n_valid` | Number of finite values used. |
| `n_missing` | `n_total - n_valid`. |
| `mean` | Arithmetic mean of the finite values. |
| `median` / `p50` | 50th percentile. Both are retained for explicit tabular semantics. |
| `std_ddof1` | Sample standard deviation with one degree of freedom. |
| `minimum` / `maximum` | Smallest and largest finite values. |
| `p05`, `p25`, `p75`, `p95` | 5th, 25th, 75th, and 95th percentiles. |
| `iqr` | Interquartile range, `p75 - p25`. |

### Composition, threshold, and comparison fields

`morphology_composition.csv` stores the class count, full-catalogue fraction,
and two-sided 95% Wilson interval. Wilson intervals were chosen because they
remain well behaved for rare classes and fractions near zero or one.

`threshold_sensitivity.csv` compares provisional `MP_LTG` thresholds of 0.5,
0.6, and 0.8 with `FLAG_LTG == 5` as an **internal operational reference**.
Its `true_positive`, `true_negative`, `false_positive`, and `false_negative`
names describe agreement with that catalogue flag; they do not describe
validation against independent morphological truth. `agreement_with_flag5` is
the fraction of threshold decisions equal to the flag-5 reference.

`highdens_minus_highlum.csv` always defines a signed difference as:

```text
highdens value - highlum value
```

Thus a positive `median_difference` or `fraction_difference` means that the
reported quantity is larger in `highdens`; a negative value means that it is
larger in `highlum`.

---

## ⚙️ Reproducibility Settings

| Constant | Value | Why it exists |
| --- | ---: | --- |
| `SEED` | `20260713` | Makes rendering samples and parent-catalogue indices reproducible; the value records the meeting date. |
| `MAX_SEPARATION_ARCSEC` | `1.0` | Enforces the adopted maximum angular cross-match tolerance. |
| `FLUX_RADIUS_CUT` | `50.0` pixels | Preserves the advisor’s provisional diagnostic. It is reported but not applied because it removes no rows in either matched product and its intended scientific role is unconfirmed. |
| `PLOT_SAMPLE_SIZE` | `100000` | Caps point-based rendering and defines the parent index-sample size without changing full-data statistics. |
| `MIN_FAINTNESS_BIN_COUNT` | `100` | Excludes unstable low-occupancy bins from the Figure 6 trend summary. |
| `OUTPUT_ROOT` | `outputs/meeting-2026-07-13/` | Keeps all generated research artifacts in one ignored, reproducible location. |

Sampling is used only where point rendering would otherwise be unnecessarily
expensive. Counts, percentiles, masks, and catalogue interpretations continue
to use all valid rows.

---

## 🛠️ Methodology: What Was Done and Why

### Memory-conscious catalogue access

The FITS files range from tens of megabytes to several gigabytes. The workflow
first inspects the table schema through memory-mapped FITS access and then
copies only the 21 required columns. It does not load every column merely to
discover the file structure. This reduces memory use and makes the scientific
dependency on each column explicit.

### Highlum-first quality gate

The smaller `highlum` matched file is read and validated before the 1.27-GB
`highdens` product is loaded. This ordering is intentional: if the known
meeting baseline has changed, the notebook stops before paying the cost of
loading the larger comparison file.

The strict `highlum` regression baseline requires 34,768 rows and the following
`FLAG_LTG` counts:

| Flag | Expected count |
| ---: | ---: |
| 0 | 5,220 |
| 1 | 3,039 |
| 2 | 344 |
| 3 | 404 |
| 4 | 24,871 |
| 5 | 890 |

An explicit runtime `ValueError`, rather than a removable Python `assert`,
enforces this gate. The `highdens` row count is recorded in provenance but is
not frozen as an approved invariant, allowing a legitimate future catalogue
replacement to pass the scientific checks.

For both matched products, the gate verifies:

- row-count consistency between the FITS schema and selected arrays;
- finite right ascension and declination inside their adopted domains;
- finite model outputs inside `[0, 1]`;
- `FLAG_LTG` values inside the expected 0–5 domain;
- nonnegative coordinate separation no larger than one arcsecond; and
- absence of duplicated morphology-side or source-side identifiers.

No invalid or duplicated rows are silently dropped. A failed gate stops the
analysis.

### Robust sample definition

The primary morphology comparison follows Vega-Ferrero et al. (2021), Table 6:

```python
robust_etg = FLAG_LTG == 4
robust_ltg = FLAG_LTG == 5
```

Flags 0–3 remain visible in the composition figure and table. This choice
avoids hiding the fraction excluded by the robust selection.

### Statistical summaries

Finite values are summarized using counts, means, medians, sample standard
deviations, fixed percentiles, and interquartile ranges. Class fractions are
accompanied by 95% Wilson intervals. The same summary objects feed both saved
tables and natural-language interpretations, preventing hard-coded prose from
becoming inconsistent with updated data.

No null-hypothesis significance test is applied. Such a test would require a
clear sampling model, known selection functions, and a treatment of dependence
between the matched samples. Those conditions are not yet established.

### Matplotlib design and artifact saving

All figures are constructed with Matplotlib. Reusable plotting functions
return `matplotlib.figure.Figure` objects and do not read data, save files, or
call `plt.show()`. The notebook is responsible for orchestration: it displays
each returned figure, saves it at 150 dpi with a stable name, and closes it to
limit memory growth.

Paired plots use common axes and bins. Figures 4 and 7 also share one
logarithmic color normalization between catalogue panels, so the same color
represents the same hexagonal-bin count in both products.

### Deterministic sampling

The sky figure uses at most 100,000 deterministic rows per matched catalogue
for point rendering. Coordinate extents and reported footprint descriptions use
all valid coordinates. The parent morphology artifact stores 100,000 unique,
sorted row **indices**, not copied catalogue values. This makes a later
extraction reproducible without duplicating a multi-gigabyte scientific input.

---

## 📊 Figure Guide and Scientific Rationale

| Figure | Scientific question | Method | Why this design was chosen |
| ---: | --- | --- | --- |
| 1 | Which morphology outcomes dominate each matched product? | Stacked count and fraction bars for robust ETG, robust LTG, and other flags. | Counts show sample scale; fractions allow comparison despite the very different catalogue sizes. |
| 2 | Do robust ETG and LTG flags occupy the expected regions of `MP_LTG`? | Normalized step histograms on common probability bins. | Normalization prevents the larger `highdens` file from dominating by raw count. |
| 3 | How much do the five LTG models disagree? | Normalized distributions of row-wise sample standard deviation. | Separates model-to-model dispersion from the median probability shown in Figure 2. |
| 4 | Where do all matched rows lie in apparent-brightness/half-light-radius space? | Paired full-data hexbins with shared axes and shared logarithmic count normalization. | Hexagonal density avoids severe scatter overplotting while preserving boundaries and outliers. |
| 5 | Do robust ETG and LTG classifications occupy different observed brightness-size loci? | Four shared-axis density panels, separated by catalogue and robust class. | Prevents abundant ETGs from hiding the smaller LTG distributions. |
| 6 | Does the distribution of `MP_LTG` change for fainter objects? | Probability-density hexbins with fixed-bin medians and interquartile bands. | Shows the complete distribution while excluding sparse bins from trend language. |
| 7 | How are morphology and viewing-orientation outputs related? | Paired `MP_LTG`–`MP_EdgeOn` hexbins with a shared logarithmic scale. | Displays the joint prediction structure without turning edge-on probability into a morphology class. |
| 8 | Do the two products occupy the same observed sky footprint? | Deterministic sampled RA/DEC scatter with shared limits. | Preserves survey gaps and masks without interpolating across unobserved areas. |
| 9 | Are coordinate-match separations comparable and within tolerance? | Normalized step histograms on a logarithmic horizontal axis, with the one-arcsecond limit marked. | Makes the narrow core and rare tail visible despite unequal file sizes. |

Every primary figure is immediately followed in the notebook by a calculated
interpretation and a separate “Careful interpretation” section.

---

## 🔬 Principal Results from the Verified Run

These values describe the local catalogues executed at Git commit `c42732c`.
They are not universal properties of DES galaxies.

### Catalogue quality

Both matched products passed the adopted quality gate:

| Catalogue | Rows | Columns | Invalid separations | Invalid coordinates | Invalid probabilities | Invalid flags | Duplicate morphology IDs | Duplicate source IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `highlum` | 34,768 | 191 | 0 | 0 | 0 | 0 | 0 | 0 |
| `highdens` | 905,291 | 191 | 0 | 0 | 0 | 0 | 0 | 0 |

### Robust composition

| Catalogue | Robust ETG | ETG fraction | Robust LTG | LTG fraction | Other flags | Other fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `highlum` | 24,871 | 71.53% | 890 | 2.56% | 9,007 | 25.91% |
| `highdens` | 369,660 | 40.83% | 62,565 | 6.91% | 473,066 | 52.26% |

The robust-LTG fraction difference, defined as `highdens - highlum`, is
approximately **+4.35 percentage points**. This difference can reflect the
upstream source selections and must not be attributed to environment without a
selection-function analysis.

### Classification probabilities and model disagreement

- Robust ETG median `MP_LTG` is 0.0015 in `highlum` and 0.0255 in `highdens`.
- Robust LTG median `MP_LTG` is 0.9689 in `highlum` and 0.9524 in `highdens`.
- The largest typical five-model disagreement occurs for `highdens` robust
  LTGs, with median `P1_P5_LTG_STD = 0.0551`.

The separation between robust ETG and LTG probabilities is partly expected
because the robust flags themselves are constructed from agreement among these
model outputs. Figure 2 is therefore an internal-consistency check, not an
independent accuracy measurement.

### Brightness and image-plane size

The central 90% `highlum` locus spans approximately `MAG_AUTO_R = 20.82–21.47`
mag and `FLUX_RADIUS_R = 3.27–5.18` pixels. Its median radius changes from 4.36
pixels in the brightest magnitude quartile to 3.91 pixels in the faintest.

The central 90% `highdens` locus spans approximately
`MAG_AUTO_R = 19.01–21.30` mag and `FLUX_RADIUS_R = 2.97–5.00` pixels. Its
corresponding median radius changes from 4.19 to 3.42 pixels.

The provisional `FLUX_RADIUS_R < 50` diagnostic removes **zero rows** from both
matched products. It is not applied to the primary results.

Within `highlum`, the robust-LTG median is 0.02 mag fainter and 0.63 pixels
larger than the robust-ETG median. Within `highdens`, it is 0.19 mag fainter and
approximately equal in median radius. These are differences in related imaging
measurements, not an independent physical morphology-size test.

### Faintness structure

Using only magnitude bins with at least 100 rows:

- `highlum` median `MP_LTG` changes from 0.006 in the brightest eligible bin
  (`n = 540`) to 0.007 in the faintest (`n = 16,660`);
- `highdens` median `MP_LTG` changes from 0.023 (`n = 1,566`) to 0.228
  (`n = 72,634`).

This is a change in model-output structure. It is not evidence of a change in
classification accuracy because the matched products do not supply independent
truth labels.

### Orientation output

In `highlum`, 82.3% of valid rows have `MP_EdgeOn <= 0.05`. Median
`MP_EdgeOn` is 0.008 when `MP_LTG <= 0.2` and 0.064 when `MP_LTG >= 0.8`.

In `highdens`, 87.2% of valid rows have `MP_EdgeOn <= 0.05`. The corresponding
medians are 0.003 and 0.011. Projection can hide disk structure, so these joint
patterns require orientation-aware interpretation.

### Sky footprint

The `highlum` matched coordinates occupy a wrapped right-ascension arc from
approximately 313.60 degrees through zero to 90.36 degrees, with declination
from -66.13 to 4.40 degrees. `highdens` occupies approximately 301.09 degrees
through zero to 97.19 degrees, with declination from -66.42 to 4.44 degrees.

Visible holes may be survey boundaries or bright-star masks. They are not
evidence of physical galaxy underdensities without an official survey mask and
selection analysis.

### Coordinate-match quality

The median, 95th percentile, and maximum separations are:

| Catalogue | Median | 95th percentile | Maximum |
| --- | ---: | ---: | ---: |
| `highlum` | 0.001286 arcsec | 0.001993 arcsec | 0.8112 arcsec |
| `highdens` | 0.001274 arcsec | 0.001989 arcsec | 0.9823 arcsec |

All rows remain inside the adopted one-arcsecond angular tolerance. The rare
tail, especially the `highdens` maximum near the limit, is retained in the
reported interpretation rather than hidden by the narrow central distribution.

---

## ⚠️ Scientific Limitations

The notebook is deliberately descriptive. Its principal limitations are:

1. The exact physical definitions and thresholds behind the `highlum` and
   `highdens` source selections remain unconfirmed.
2. The two files have different sizes and selection mixtures; normalized plots
   improve visual comparability but do not equalize their selection functions.
3. `MAG_AUTO_R` is apparent magnitude, and `FLUX_RADIUS_R` is a pixel radius.
   Neither quantity is a distance-corrected physical observable.
4. The CNN outputs are not independent truth labels or guaranteed calibrated
   frequencies.
5. The five models may share data, architecture, and systematic biases; their
   disagreement is not total uncertainty.
6. Sky gaps cannot be interpreted physically without the official footprint
   and masks.
7. The analysis lacks the corrected morphometric parameters discussed with
   Fabrício, so the Ferrari et al. morphometric analysis remains deferred.
8. No causal environmental inference is supported at this stage.

---

## 🧪 Reproducibility and Software Organization

Reusable work is separated from notebook narration:

| Module | Responsibility |
| --- | --- |
| `src/galaxy_analysis/catalog.py` | FITS schema inspection, selected-column reading, and deterministic indices. |
| `src/galaxy_analysis/selection.py` | Scientific validity domains, angular-match validation, and robust masks. |
| `src/galaxy_analysis/quality.py` | Reusable pass/fail catalogue gates and the strict `highlum` baseline. |
| `src/galaxy_analysis/statistics.py` | Composition, Wilson intervals, descriptive summaries, thresholds, model dispersion, and catalogue differences. |
| `src/galaxy_analysis/plotting.py` | Pure Matplotlib constructors for the nine primary figures. |
| `src/galaxy_analysis/reporting.py` | Data-dependent, scientifically cautious result paragraphs. |
| `src/galaxy_analysis/artifacts.py` | Project discovery and stable figure, JSON, and text output. |

This separation keeps the notebook readable and lets unit tests verify the
scientific definitions. The tracked `.ipynb` contains no execution counts or
outputs. A clean-kernel verification executes a temporary copy, preserving a
reviewable Git diff.

The verified software environment used Python 3.13.2, NumPy 2.5.1, Astropy
8.0.1, and Matplotlib 3.11.0. `run_metadata.json` records the exact versions,
input sizes and timestamps, Git commit, dirty-tree status, parameters,
warnings, and product names for each execution.

---

## 🚀 Getting Started

### Prepare the environment

From the repository root:

```bash
python -m venv scientific-research
source scientific-research/bin/activate
python -m pip install -r requirements.txt
```

Place the local FITS catalogues in the paths documented in
[`../data/README.md`](../data/README.md). The FITS files are scientific inputs
and must remain excluded from Git.

### Interactive execution

```bash
jupyter lab notebooks/01_robust_morphology_environment_analysis.ipynb
```

Run cells from top to bottom. If Jupyter is launched outside the repository,
set `GALAXY_PROJECT_ROOT` to the repository root before starting it.

### Clean-kernel verification without modifying the tracked notebook

```bash
GALAXY_PROJECT_ROOT="$PWD" \
MPLBACKEND=Agg \
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=1800 \
  --output /tmp/robust-morphology-executed.ipynb \
  notebooks/01_robust_morphology_environment_analysis.ipynb
```

Run the tests separately:

```bash
MPLBACKEND=Agg python -m pytest -q
```

Before committing, verify that the tracked notebook is still output-free and
that no FITS file or generated artifact is staged.

---

## 📁 Generated Artifacts

Each execution recreates the following ignored products below
`outputs/meeting-2026-07-13/`:

```text
outputs/meeting-2026-07-13/
├── advisor_summary.md
├── catalogue_quality.json
├── run_metadata.json
├── figures/
│   ├── 01_catalogue_composition.png
│   ├── 02_robust_ltg_probability.png
│   ├── 03_model_disagreement.png
│   ├── 04_magnitude_half_light_radius.png
│   ├── 05_morphology_brightness_size.png
│   ├── 06_probability_vs_faintness.png
│   ├── 07_ltg_vs_edgeon_probability.png
│   ├── 08_sky_footprint.png
│   └── 09_match_separation.png
├── parent-sample/
│   └── sample_indices.npy
└── tables/
    ├── catalogue_quality.csv
    ├── highdens_minus_highlum.csv
    ├── morphology_composition.csv
    ├── summary_statistics.csv
    └── threshold_sensitivity.csv
```

These files are reproducible research products, not source files. They remain
ignored so the repository stays clean and does not accumulate generated binary
figures, execution-specific metadata, or large extracted data.

---

## 📌 Questions for the Next Advisor Meeting

1. What exact luminosity, density, redshift, and membership rules define the
   upstream `highlum` and `highdens` source selections?
2. Was `FLUX_RADIUS_R < 50` intended as a scientific filter, an axis limit, or
   only a diagnostic of the parent catalogue?
3. Should thresholds 0.5, 0.6, and 0.8 remain sensitivity checks, or should one
   support a future alternate sample definition?
4. Can the corrected catalogue containing the missing morphometric parameters
   be provided?
5. Should a later phase convert image-plane radii to physical radii using
   redshift, pixel scale, and cosmology?
6. Which statistical comparison is appropriate after the selection functions
   and dependence between samples are understood?

---

## 📚 Foundational Literature and References

- Cheng, T.-Y., et al. (2021). *Galaxy Morphological Classification Catalogue
  of the Dark Energy Survey Year 3 Data with Convolutional Neural Networks*.
- Ferrari, F., de Carvalho, R. R., & Trevisan, M. (2015). *Morfometryka—A New
  Way of Establishing Morphological Classification of Galaxies*. The
  Astrophysical Journal, 814, 55.
- Vega-Ferrero, J., et al. (2021). *Pushing Automated Morphological
  Classifications to Their Limits with the Dark Energy Survey*. The catalogue
  definitions used here follow Table 6.

---

## 📍 Scope of the Conclusions

The notebook establishes a validated and reproducible description of the two
supplied morphology cross-matches. It does not yet establish why the catalogue
distributions differ. That distinction—between a reproducible measured
association and a physical causal explanation—is the central methodological
reason for the workflow documented here.

*Viva a ciência brasileira!* 🇧🇷🔬
