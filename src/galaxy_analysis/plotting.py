"""Pure Matplotlib figure constructors for galaxy morphology analysis.

These functions never read FITS files or save output.  They only transform
supplied arrays into figures, leaving the notebook to choose reproducible input
samples, stable filenames, and output directories.  This separation is why the
scientific encodings can be tested with tiny synthetic arrays.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .selection import MorphologyMasks
from .statistics import FlagCountRow


ETG_COLOR = "tab:red"
LTG_COLOR = "tab:blue"


def _finite_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return rows where both plotted variables are finite numeric values."""

    return np.isfinite(np.asarray(first, dtype=float)) & np.isfinite(
        np.asarray(second, dtype=float)
    )


def _finish(figure: Figure) -> Figure:
    """Apply the common anti-clipping layout and return the same figure."""

    figure.tight_layout()
    return figure


def plot_magnitude_radius_scatter(
    magnitude: np.ndarray,
    radius: np.ndarray,
    masks: MorphologyMasks | None,
    catalog_alias: str,
    flux_radius_max: float | None = None,
) -> Figure:
    """Plot magnitude against angular radius for all rows or robust classes.

    ``masks=None`` is the unbiased coverage view.  Passing morphology masks
    produces the flag-4/flag-5 comparison without reclassifying flags 0–3.  An
    optional radius limit supports the advisor's provisional ``<50`` request.
    """
    magnitude = np.asarray(magnitude, dtype=float)
    radius = np.asarray(radius, dtype=float)
    valid = _finite_pair(magnitude, radius)
    if flux_radius_max is not None:
        valid &= radius < flux_radius_max

    figure, axis = plt.subplots(figsize=(8, 6))
    if masks is None:
        axis.scatter(
            magnitude[valid],
            radius[valid],
            s=8,
            alpha=0.3,
            color="0.25",
            rasterized=True,
            label=f"All valid rows (N={np.count_nonzero(valid):,})",
        )
    else:
        for class_mask, label, color in (
            (masks.robust_etg, "Robust ETG", ETG_COLOR),
            (masks.robust_ltg, "Robust LTG", LTG_COLOR),
        ):
            selected = valid & np.asarray(class_mask, dtype=bool)
            axis.scatter(
                magnitude[selected],
                radius[selected],
                s=8,
                alpha=0.3,
                color=color,
                # Rasterization keeps dense points compact inside vector-like
                # notebook output without changing labels or axes.
                rasterized=True,
                label=f"{label} (N={np.count_nonzero(selected):,})",
            )
    suffix = "" if flux_radius_max is None else f"; FLUX_RADIUS_R < {flux_radius_max:g}"
    sample_name = "all valid rows" if masks is None else "robust morphology"
    axis.set_title(f"{catalog_alias}: {sample_name} magnitude–radius{suffix}")
    axis.set_xlabel("MAG_AUTO_R [mag] (brighter ←)")
    axis.set_ylabel("FLUX_RADIUS_R [pixel]")
    # Astronomical magnitude runs backwards: smaller numbers are brighter.
    axis.invert_xaxis()
    axis.legend()
    axis.grid(alpha=0.2)
    return _finish(figure)


def plot_magnitude_radius_density(
    magnitude: np.ndarray,
    radius: np.ndarray,
    catalog_alias: str,
    flux_radius_max: float | None = None,
) -> Figure:
    """Plot the full valid magnitude–radius density as logarithmic hexagons.

    Hexbin uses every valid row and reveals the dense locus without the severe
    overplotting of a million-point scatter plot.  Log counts retain visibility
    for both common and sparse regions.
    """
    magnitude = np.asarray(magnitude, dtype=float)
    radius = np.asarray(radius, dtype=float)
    valid = _finite_pair(magnitude, radius)
    if flux_radius_max is not None:
        valid &= radius < flux_radius_max

    figure, axis = plt.subplots(figsize=(8, 6))
    density = axis.hexbin(
        magnitude[valid], radius[valid], gridsize=70, bins="log", mincnt=1, cmap="viridis"
    )
    colorbar = figure.colorbar(density, ax=axis)
    colorbar.set_label("logarithmic count per hexagonal bin")
    axis.set_title(f"{catalog_alias}: magnitude–radius density (N={np.count_nonzero(valid):,})")
    axis.set_xlabel("MAG_AUTO_R [mag] (brighter ←)")
    axis.set_ylabel("FLUX_RADIUS_R [pixel]")
    axis.invert_xaxis()
    return _finish(figure)


def plot_probability_by_class(
    probability: np.ndarray,
    masks: MorphologyMasks,
    probability_name: str,
    catalog_alias: str,
) -> Figure:
    """Compare normalized probability distributions for robust ETGs and LTGs.

    Identical bins and density normalization compare distribution shape despite
    the strong class imbalance.  Fixed probability bounds prevent autoscaling
    from making two model outputs appear less comparable than they are.
    """
    probability = np.asarray(probability, dtype=float)
    valid = np.isfinite(probability) & (probability >= 0) & (probability <= 1)
    bins = np.linspace(0, 1, 41)
    figure, axis = plt.subplots(figsize=(8, 5))
    for class_mask, label, color in (
        (masks.robust_etg, "Robust ETG", ETG_COLOR),
        (masks.robust_ltg, "Robust LTG", LTG_COLOR),
    ):
        selected = valid & np.asarray(class_mask, dtype=bool)
        values = probability[selected]
        axis.hist(
            values,
            bins=bins,
            density=bool(values.size),
            histtype="step",
            linewidth=2,
            color=color,
            label=f"{label} (N={values.size:,})",
        )
    axis.set_xlim(0, 1)
    axis.set_title(f"{catalog_alias}: {probability_name} by robust class")
    axis.set_xlabel(f"{probability_name} probability")
    axis.set_ylabel("Normalized density")
    axis.legend()
    axis.grid(alpha=0.2)
    return _finish(figure)


def plot_model_dispersion_by_class(
    dispersion: np.ndarray,
    masks: MorphologyMasks,
    catalog_alias: str,
) -> Figure:
    """Show per-object P1–P5 LTG dispersion with median and IQR by class.

    A box plot emphasizes the median and middle 50% rather than allowing a
    small number of extreme disagreements to dominate the vertical scale.
    """
    dispersion = np.asarray(dispersion, dtype=float)
    groups = []
    for class_mask in (masks.robust_etg, masks.robust_ltg):
        selected = np.isfinite(dispersion) & np.asarray(class_mask, dtype=bool)
        groups.append(dispersion[selected])

    figure, axis = plt.subplots(figsize=(7, 5))
    boxes = axis.boxplot(
        groups,
        tick_labels=[f"Robust ETG\nN={groups[0].size:,}", f"Robust LTG\nN={groups[1].size:,}"],
        # Outliers remain in numerical summaries; hiding their markers keeps
        # this diagnostic focused on the requested median and IQR.
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(boxes["boxes"], (ETG_COLOR, LTG_COLOR), strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    axis.set_title(f"{catalog_alias}: five-model LTG probability dispersion")
    axis.set_xlabel("Robust morphology class")
    axis.set_ylabel("P1–P5 LTG sample standard deviation")
    axis.grid(axis="y", alpha=0.2)
    return _finish(figure)


def plot_probability_vs_magnitude(
    magnitude: np.ndarray,
    probability: np.ndarray,
    catalog_alias: str,
) -> Figure:
    """Plot LTG probability density against apparent magnitude.

    The magnitude axis increases normally here so moving right explicitly means
    fainter sources.  This makes it easier to inspect—not causally attribute—
    growing probability ambiguity toward lower apparent flux.
    """
    magnitude = np.asarray(magnitude, dtype=float)
    probability = np.asarray(probability, dtype=float)
    valid = _finite_pair(magnitude, probability) & (probability >= 0) & (probability <= 1)
    figure, axis = plt.subplots(figsize=(8, 6))
    density = axis.hexbin(
        magnitude[valid],
        probability[valid],
        gridsize=70,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    figure.colorbar(density, ax=axis, label="logarithmic count per hexagonal bin")
    axis.set_ylim(0, 1)
    axis.set_title(
        f"{catalog_alias}: LTG probability versus apparent magnitude "
        f"(N={np.count_nonzero(valid):,})"
    )
    axis.set_xlabel("MAG_AUTO_R [mag] (fainter →)")
    axis.set_ylabel("MP_LTG probability")
    return _finish(figure)


def plot_edgeon_vs_ltg_probability(
    ltg_probability: np.ndarray,
    edgeon_probability: np.ndarray,
    catalog_alias: str,
) -> Figure:
    """Plot joint density of LTG and edge-on probability estimates.

    Both outputs share fixed ``[0, 1]`` axes.  A two-dimensional density view
    exposes orientation-related branches that separate histograms would hide.
    """
    ltg_probability = np.asarray(ltg_probability, dtype=float)
    edgeon_probability = np.asarray(edgeon_probability, dtype=float)
    valid = _finite_pair(ltg_probability, edgeon_probability)
    valid &= (ltg_probability >= 0) & (ltg_probability <= 1)
    valid &= (edgeon_probability >= 0) & (edgeon_probability <= 1)
    figure, axis = plt.subplots(figsize=(7, 6))
    density = axis.hexbin(
        ltg_probability[valid],
        edgeon_probability[valid],
        gridsize=70,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    figure.colorbar(density, ax=axis, label="logarithmic count per hexagonal bin")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title(
        f"{catalog_alias}: edge-on versus LTG probability "
        f"(N={np.count_nonzero(valid):,})"
    )
    axis.set_xlabel("MP_LTG probability")
    axis.set_ylabel("MP_EdgeOn probability")
    return _finish(figure)


def plot_sky_distribution(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    masks: MorphologyMasks | None,
    catalog_alias: str,
) -> Figure:
    """Plot catalogue coordinates without interpolating across survey gaps.

    Direct rasterized points preserve holes from footprint or masking.  The
    function deliberately avoids smoothing because that could visually fill
    regions where the catalogue contains no observations.
    """
    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    valid = _finite_pair(ra_deg, dec_deg)
    figure, axis = plt.subplots(figsize=(9, 5))
    if masks is None:
        axis.scatter(
            ra_deg[valid],
            dec_deg[valid],
            s=2,
            alpha=0.3,
            color="0.25",
            rasterized=True,
            label=f"All valid coordinates (N={np.count_nonzero(valid):,})",
        )
    else:
        for class_mask, label, color in (
            (masks.robust_etg, "Robust ETG", ETG_COLOR),
            (masks.robust_ltg, "Robust LTG", LTG_COLOR),
        ):
            selected = valid & np.asarray(class_mask, dtype=bool)
            axis.scatter(
                ra_deg[selected],
                dec_deg[selected],
                s=3,
                alpha=0.35,
                color=color,
                rasterized=True,
                label=f"{label} (N={np.count_nonzero(selected):,})",
            )
    axis.set_title(f"{catalog_alias}: observed sky positions")
    axis.set_xlabel("Right ascension RA_2 [deg]")
    axis.set_ylabel("Declination DEC_2 [deg]")
    axis.legend(markerscale=3)
    axis.grid(alpha=0.15)
    return _finish(figure)


def plot_flag_counts(flags: np.ndarray, catalog_alias: str) -> Figure:
    """Plot counts and percentages for all ``FLAG_LTG`` values 0 through 5.

    Neutral colors keep flags 0–3 visibly non-robust; only the explicitly used
    flag-4 ETG and flag-5 LTG bars receive the analysis class colors.
    """
    flags = np.asarray(flags)
    counts = np.array([np.count_nonzero(flags == value) for value in range(6)])
    total = int(flags.size)
    colors = ["0.65", "0.65", "0.65", "0.65", ETG_COLOR, LTG_COLOR]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(np.arange(6), counts, color=colors)
    for bar, count in zip(bars, counts, strict=True):
        percentage = 100 * count / total if total else 0.0
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(np.arange(6), [str(value) for value in range(6)])
    axis.set_title(f"{catalog_alias}: FLAG_LTG composition (N={total:,})")
    axis.set_xlabel("FLAG_LTG value")
    axis.set_ylabel("Galaxy count")
    axis.grid(axis="y", alpha=0.2)
    return _finish(figure)


def plot_robust_fraction_comparison(
    highlum_counts: list[FlagCountRow],
    highdens_counts: list[FlagCountRow],
) -> Figure:
    """Compare robust class fractions with 95% Wilson interval error bars.

    Fractions use each complete matched catalogue as denominator.  Wilson
    intervals communicate counting precision while the notebook separately
    warns that selection-systematic uncertainty can be much larger.
    """
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(2)
    width = 0.36
    for offset, rows, label, color in (
        (-width / 2, highlum_counts, "highlum", "0.35"),
        (width / 2, highdens_counts, "highdens", "tab:green"),
    ):
        by_flag = {row.flag_ltg: row for row in rows}
        selected = [by_flag[4], by_flag[5]]
        fractions = np.array([row.fraction_of_total for row in selected])
        lower = np.array([
            fraction - (row.wilson_low_95 if row.wilson_low_95 is not None else fraction)
            for fraction, row in zip(fractions, selected, strict=True)
        ])
        upper = np.array([
            (row.wilson_high_95 if row.wilson_high_95 is not None else fraction) - fraction
            for fraction, row in zip(fractions, selected, strict=True)
        ])
        axis.bar(
            positions + offset,
            fractions,
            width,
            yerr=np.vstack([lower, upper]),
            capsize=4,
            color=color,
            alpha=0.8,
            label=label,
        )
    axis.set_xticks(positions, ["Robust ETG (flag 4)", "Robust LTG (flag 5)"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Fraction of matched catalogue")
    axis.set_title("Robust morphology fractions with Wilson 95% intervals")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    return _finish(figure)


def plot_parameter_comparison(
    variable: str,
    highlum_values: Mapping[str, np.ndarray],
    highdens_values: Mapping[str, np.ndarray],
) -> Figure:
    """Compare normalized parameter distributions on shared class-panel limits.

    Separate ETG/LTG panels control class mixture, normalization controls
    catalogue size, and pooled limits prevent autoscaling from manufacturing a
    visual difference between ``highlum`` and ``highdens``.
    """
    class_names = ("robust_etg", "robust_ltg")
    finite_arrays = []
    for values_by_class in (highlum_values, highdens_values):
        for class_name in class_names:
            values = np.asarray(values_by_class[class_name], dtype=float)
            finite_arrays.append(values[np.isfinite(values)])
    pooled = np.concatenate([values for values in finite_arrays if values.size])
    # Probability outputs have a physical/model domain known in advance.  Other
    # variables use a single pooled domain shared by every line and panel.
    if variable.startswith("MP_"):
        limits = (0.0, 1.0)
    elif pooled.size:
        limits = (float(np.min(pooled)), float(np.max(pooled)))
        if limits[0] == limits[1]:
            limits = (limits[0] - 0.5, limits[1] + 0.5)
    else:
        limits = (0.0, 1.0)
    bins = np.linspace(*limits, 41)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    for axis, class_name, class_label in zip(
        axes, class_names, ("Robust ETG (flag 4)", "Robust LTG (flag 5)"), strict=True
    ):
        for values_by_class, catalog_alias, color in (
            (highlum_values, "highlum", "0.25"),
            (highdens_values, "highdens", "tab:green"),
        ):
            values = np.asarray(values_by_class[class_name], dtype=float)
            values = values[np.isfinite(values)]
            axis.hist(
                values,
                bins=bins,
                density=bool(values.size),
                histtype="step",
                linewidth=2,
                color=color,
                label=catalog_alias,
            )
        axis.set_xlim(limits)
        axis.set_xlabel(variable)
        axis.set_ylabel("Normalized density")
        axis.set_title(class_label)
        axis.legend()
        axis.grid(alpha=0.2)
    figure.suptitle(f"Matched-catalogue comparison: {variable}")
    return _finish(figure)


def plot_separation_comparison(
    highlum_separation: np.ndarray,
    highdens_separation: np.ndarray,
    max_arcsec: float = 1.0,
) -> Figure:
    """Compare normalized coordinate-match separations on a logarithmic axis.

    Most valid matches are near ``1e-3`` arcsec while the acceptance rule is
    1 arcsec.  Logarithmic bins make the narrow core readable without hiding the
    full threshold.  Exact zeros, which cannot appear on a log axis, are kept in
    the first positive-width bin instead of being discarded.
    """
    figure, axis = plt.subplots(figsize=(8, 5))
    prepared = []
    for source in (highlum_separation, highdens_separation):
        values = np.asarray(source, dtype=float)
        prepared.append(values[np.isfinite(values) & (values >= 0) & (values <= max_arcsec)])
    # One lower bound is derived from both catalogues so neither receives a
    # more favorable resolution or axis range.
    positive = np.concatenate([values[values > 0] for values in prepared])
    lower = float(np.min(positive)) if positive.size else max_arcsec * 1e-6
    bins = np.geomspace(lower, max_arcsec, 51)
    for values, label, color in (
        (prepared[0], "highlum", "0.25"),
        (prepared[1], "highdens", "tab:green"),
    ):
        plotted = np.where(values == 0, lower, values)
        axis.hist(
            plotted,
            bins=bins,
            density=bool(plotted.size),
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
        )
    axis.axvline(max_arcsec, color="tab:red", linestyle="--", label=f"{max_arcsec:g} arcsec limit")
    axis.set_xscale("log")
    axis.set_xlim(lower, max_arcsec * 1.05)
    axis.set_xlabel("Coordinate-match separation [arcsec, log scale]")
    axis.set_ylabel("Normalized density")
    axis.set_title("Coordinate-match separation comparison")
    axis.legend()
    axis.grid(alpha=0.2)
    return _finish(figure)
