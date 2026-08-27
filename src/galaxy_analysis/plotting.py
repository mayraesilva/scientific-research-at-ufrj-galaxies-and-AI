"""Pure Matplotlib figures for the matched-catalogue research narrative.

Each constructor answers one notebook question and compares ``highlum`` with
``highdens`` directly.  The functions neither read FITS files nor save output;
the notebook owns input provenance, deterministic sampling, display, and file
writes.  Keeping those responsibilities separate makes every visual encoding
testable with small arrays.
"""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .selection import MorphologyMasks
from .statistics import CompositionRow, binned_quantiles


ETG_COLOR = "tab:red"
LTG_COLOR = "tab:blue"
OTHER_COLOR = "0.65"
HIGHLUM_COLOR = "0.25"
HIGHDENS_COLOR = "tab:green"

CLASS_NAMES = ("robust_etg", "robust_ltg", "other_flags")
CLASS_LABELS = ("Robust ETG", "Robust LTG", "Other flags")
CLASS_COLORS = (ETG_COLOR, LTG_COLOR, OTHER_COLOR)


def _finite_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return a mask for rows where both plotted quantities are finite."""

    return np.isfinite(np.asarray(first, dtype=float)) & np.isfinite(
        np.asarray(second, dtype=float)
    )


def _composition_matrix(rows: list[CompositionRow], field: str) -> np.ndarray:
    """Return grouped composition values in the fixed scientific class order."""

    by_class = {row.class_name: row for row in rows}
    missing = set(CLASS_NAMES) - set(by_class)
    if missing:
        raise ValueError(f"composition rows are missing {sorted(missing)}")
    return np.array([getattr(by_class[name], field) for name in CLASS_NAMES])


def _finite_values(values: np.ndarray) -> np.ndarray:
    """Return finite floating-point values for normalized distributions."""

    numeric = np.asarray(values, dtype=float)
    return numeric[np.isfinite(numeric)]


def _pooled_limits(*arrays: np.ndarray) -> tuple[float, float]:
    """Return one finite plotting range shared by every comparison panel."""

    pooled = np.concatenate([_finite_values(values) for values in arrays])
    if not pooled.size:
        raise ValueError("shared plot limits require finite values")

    lower = float(pooled.min())
    upper = float(pooled.max())
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.5)
        return lower - padding, upper + padding
    return lower, upper


def plot_catalogue_composition(
    highlum_rows: list[CompositionRow],
    highdens_rows: list[CompositionRow],
) -> Figure:
    """Compare class counts and fractions without hiding unequal file sizes.

    The count panel communicates workload and sample imbalance.  The fraction
    panel makes the two catalogues comparable despite their different row
    totals.  Stacked class colors retain one meaning throughout the notebook.
    """

    count_matrix = np.vstack(
        [
            _composition_matrix(highlum_rows, "count"),
            _composition_matrix(highdens_rows, "count"),
        ]
    )
    fraction_matrix = np.vstack(
        [
            _composition_matrix(highlum_rows, "fraction_of_total"),
            _composition_matrix(highdens_rows, "fraction_of_total"),
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    positions = np.arange(2)
    catalog_labels = ("highlum", "highdens")

    for axis, matrix, ylabel in zip(
        axes,
        (count_matrix, fraction_matrix),
        ("Galaxy count", "Fraction of matched catalogue"),
        strict=True,
    ):
        bottom = np.zeros(2)
        for class_index, (class_label, color) in enumerate(
            zip(CLASS_LABELS, CLASS_COLORS, strict=True)
        ):
            heights = matrix[:, class_index]
            axis.bar(
                positions,
                heights,
                bottom=bottom,
                color=color,
                label=class_label,
            )
            bottom += heights
        axis.set_xticks(positions, catalog_labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)

    axes[0].legend()
    axes[1].set_ylim(0, 1)
    figure.suptitle("Morphology composition of the matched catalogues")
    return figure


def plot_robust_probability_comparison(
    highlum_values: Mapping[str, np.ndarray],
    highdens_values: Mapping[str, np.ndarray],
) -> Figure:
    """Compare normalized MP_LTG distributions within each robust class."""

    bins = np.linspace(0, 1, 41)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for axis, class_name, title in zip(
        axes,
        ("robust_etg", "robust_ltg"),
        ("Robust ETG (flag 4)", "Robust LTG (flag 5)"),
        strict=True,
    ):
        for values_by_class, catalog, color in (
            (highlum_values, "highlum", HIGHLUM_COLOR),
            (highdens_values, "highdens", HIGHDENS_COLOR),
        ):
            values = _finite_values(values_by_class[class_name])
            axis.hist(
                values,
                bins=bins,
                density=bool(values.size),
                histtype="step",
                linewidth=2,
                color=color,
                label=catalog,
            )
        axis.set_xlim(0, 1)
        axis.set_xlabel("MP_LTG")
        axis.set_ylabel("Normalized density")
        axis.set_title(title)
        axis.legend()
        axis.grid(alpha=0.2)

    figure.suptitle("Robust morphology classification probabilities")
    return figure


def plot_model_disagreement_comparison(
    highlum_values: Mapping[str, np.ndarray],
    highdens_values: Mapping[str, np.ndarray],
) -> Figure:
    """Compare P1-P5 LTG dispersion within each robust morphology class."""

    pooled = np.concatenate(
        [
            _finite_values(values[class_name])
            for values in (highlum_values, highdens_values)
            for class_name in ("robust_etg", "robust_ltg")
        ]
    )
    # Include the complete finite range. Excluding the tail could leave a
    # small robust-class histogram empty and would hide real disagreement.
    upper = float(pooled.max()) if pooled.size else 1.0
    upper = max(upper, np.finfo(float).eps)
    bins = np.linspace(0, upper, 41)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for axis, class_name, title in zip(
        axes,
        ("robust_etg", "robust_ltg"),
        ("Robust ETG (flag 4)", "Robust LTG (flag 5)"),
        strict=True,
    ):
        for values_by_class, catalog, color in (
            (highlum_values, "highlum", HIGHLUM_COLOR),
            (highdens_values, "highdens", HIGHDENS_COLOR),
        ):
            values = _finite_values(values_by_class[class_name])
            axis.hist(
                values,
                bins=bins,
                density=bool(values.size),
                histtype="step",
                linewidth=2,
                color=color,
                label=catalog,
            )
        axis.set_xlabel("P1-P5 LTG sample standard deviation")
        axis.set_ylabel("Normalized density")
        axis.set_title(title)
        axis.legend()
        axis.grid(alpha=0.2)

    figure.suptitle("Disagreement among the five LTG models")
    return figure


def plot_magnitude_radius_comparison(
    highlum: Mapping[str, np.ndarray],
    highdens: Mapping[str, np.ndarray],
) -> Figure:
    """Show full brightness-radius density on shared astronomical axes."""

    magnitude_limits = _pooled_limits(
        highlum["MAG_AUTO_R"],
        highdens["MAG_AUTO_R"],
    )
    radius_limits = _pooled_limits(
        highlum["FLUX_RADIUS_R"],
        highdens["FLUX_RADIUS_R"],
    )
    extent = (*magnitude_limits, *radius_limits)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    density = None
    for axis, data, catalog in zip(
        axes,
        (highlum, highdens),
        ("highlum", "highdens"),
        strict=True,
    ):
        magnitude = np.asarray(data["MAG_AUTO_R"], dtype=float)
        radius = np.asarray(data["FLUX_RADIUS_R"], dtype=float)
        valid = _finite_pair(magnitude, radius) & (radius > 0)
        density = axis.hexbin(
            magnitude[valid],
            radius[valid],
            gridsize=70,
            bins="log",
            mincnt=1,
            extent=extent,
            cmap="viridis",
        )
        axis.set_title(catalog)
        axis.set_xlabel("MAG_AUTO_R [mag] (brighter ←)")
        axis.set_ylabel("FLUX_RADIUS_R [pixel]")

    axes[0].invert_xaxis()
    figure.colorbar(
        density,
        ax=axes,
        label="Logarithmic count per hexagonal bin",
    )
    figure.suptitle("Apparent magnitude and half-light radius")
    return figure


def plot_morphology_brightness_size_comparison(
    highlum: Mapping[str, np.ndarray],
    highlum_masks: MorphologyMasks,
    highdens: Mapping[str, np.ndarray],
    highdens_masks: MorphologyMasks,
) -> Figure:
    """Compare robust-class brightness-radius densities on shared scales."""

    magnitude_limits = _pooled_limits(
        highlum["MAG_AUTO_R"],
        highdens["MAG_AUTO_R"],
    )
    radius_limits = _pooled_limits(
        highlum["FLUX_RADIUS_R"],
        highdens["FLUX_RADIUS_R"],
    )
    extent = (*magnitude_limits, *radius_limits)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    catalogue_rows = (
        (highlum, highlum_masks, "highlum"),
        (highdens, highdens_masks, "highdens"),
    )
    class_columns = (
        ("robust_etg", "Robust ETG", "Reds"),
        ("robust_ltg", "Robust LTG", "Blues"),
    )
    for row, (data, masks, catalog) in enumerate(catalogue_rows):
        magnitude = np.asarray(data["MAG_AUTO_R"], dtype=float)
        radius = np.asarray(data["FLUX_RADIUS_R"], dtype=float)
        finite = _finite_pair(magnitude, radius) & (radius > 0)

        for column, (mask_name, class_label, color_map) in enumerate(
            class_columns
        ):
            class_mask = np.asarray(getattr(masks, mask_name), dtype=bool)
            selected = finite & class_mask
            axis = axes[row, column]
            axis.hexbin(
                magnitude[selected],
                radius[selected],
                gridsize=55,
                bins="log",
                mincnt=1,
                extent=extent,
                cmap=color_map,
            )
            axis.set_title(f"{catalog}: {class_label}")
            axis.set_xlabel("MAG_AUTO_R [mag] (brighter ←)")
            axis.set_ylabel("FLUX_RADIUS_R [pixel]")

    axes[0, 0].invert_xaxis()
    figure.suptitle("Robust morphology in brightness-size space")
    return figure


def plot_probability_faintness_comparison(
    highlum: Mapping[str, np.ndarray],
    highdens: Mapping[str, np.ndarray],
    magnitude_bins: np.ndarray,
) -> Figure:
    """Overlay binned MP_LTG summaries on each magnitude-density panel."""

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for axis, data, catalog in zip(
        axes,
        (highlum, highdens),
        ("highlum", "highdens"),
        strict=True,
    ):
        magnitude = np.asarray(data["MAG_AUTO_R"], dtype=float)
        probability = np.asarray(data["MP_LTG"], dtype=float)
        valid = _finite_pair(magnitude, probability)
        valid &= (probability >= 0) & (probability <= 1)

        axis.hexbin(
            magnitude[valid],
            probability[valid],
            gridsize=65,
            bins="log",
            mincnt=1,
            cmap="Greys",
            alpha=0.65,
        )
        summary = binned_quantiles(magnitude, probability, magnitude_bins)
        populated = summary.count > 0
        axis.fill_between(
            summary.bin_centers[populated],
            summary.p25[populated],
            summary.p75[populated],
            color="tab:orange",
            alpha=0.25,
            label="Interquartile range",
        )
        axis.plot(
            summary.bin_centers[populated],
            summary.median[populated],
            color="tab:orange",
            linewidth=2,
            label="Binned median",
        )
        axis.set_ylim(0, 1)
        axis.set_title(catalog)
        axis.set_xlabel("MAG_AUTO_R [mag] (fainter →)")
        axis.set_ylabel("MP_LTG")
        axis.legend()

    figure.suptitle("LTG model output across apparent magnitude")
    return figure


def plot_orientation_comparison(
    highlum: Mapping[str, np.ndarray],
    highdens: Mapping[str, np.ndarray],
) -> Figure:
    """Compare joint LTG and edge-on model-output densities."""

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    density = None
    for axis, data, catalog in zip(
        axes,
        (highlum, highdens),
        ("highlum", "highdens"),
        strict=True,
    ):
        ltg_probability = np.asarray(data["MP_LTG"], dtype=float)
        edgeon_probability = np.asarray(data["MP_EdgeOn"], dtype=float)
        valid = _finite_pair(ltg_probability, edgeon_probability)
        valid &= (ltg_probability >= 0) & (ltg_probability <= 1)
        valid &= (edgeon_probability >= 0) & (edgeon_probability <= 1)

        density = axis.hexbin(
            ltg_probability[valid],
            edgeon_probability[valid],
            gridsize=65,
            bins="log",
            mincnt=1,
            extent=(0, 1, 0, 1),
            cmap="cividis",
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_title(catalog)
        axis.set_xlabel("MP_LTG")
        axis.set_ylabel("MP_EdgeOn")

    figure.colorbar(
        density,
        ax=axes,
        label="Logarithmic count per hexagonal bin",
    )
    figure.suptitle("Morphology and viewing-orientation model outputs")
    return figure


def plot_sky_footprint_comparison(
    highlum: Mapping[str, np.ndarray],
    highdens: Mapping[str, np.ndarray],
) -> Figure:
    """Show observed coordinates directly without filling catalogue gaps."""

    ra_limits = _pooled_limits(highlum["RA_2"], highdens["RA_2"])
    dec_limits = _pooled_limits(highlum["DEC_2"], highdens["DEC_2"])
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for axis, data, catalog in zip(
        axes,
        (highlum, highdens),
        ("highlum", "highdens"),
        strict=True,
    ):
        right_ascension = np.asarray(data["RA_2"], dtype=float)
        declination = np.asarray(data["DEC_2"], dtype=float)
        valid = _finite_pair(right_ascension, declination)
        axis.scatter(
            right_ascension[valid],
            declination[valid],
            s=0.3,
            alpha=0.3,
            color="midnightblue",
            edgecolors="none",
            rasterized=True,
        )
        axis.set_xlim(ra_limits)
        axis.set_ylim(dec_limits)
        axis.set_title(catalog)
        axis.set_xlabel("Right ascension [deg]")
        axis.set_ylabel("Declination [deg]")

    figure.suptitle("Observed sky footprint of each matched catalogue")
    return figure


def plot_separation_comparison(
    highlum_separation: np.ndarray,
    highdens_separation: np.ndarray,
    max_arcsec: float = 1.0,
) -> Figure:
    """Compare normalized coordinate-match separations on a logarithmic axis.

    Exact zeros cannot appear on a logarithmic axis, so they are retained in
    the first positive-width bin.  Both files use the same bins and adopted
    tolerance, preventing unequal catalogue sizes from shaping the comparison.
    """

    prepared = []
    for source in (highlum_separation, highdens_separation):
        values = np.asarray(source, dtype=float)
        valid = (
            np.isfinite(values)
            & (values >= 0)
            & (values <= max_arcsec)
        )
        prepared.append(values[valid])

    positive = np.concatenate([values[values > 0] for values in prepared])
    lower = float(positive.min()) if positive.size else max_arcsec * 1e-6
    bins = np.geomspace(lower, max_arcsec, 51)
    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")

    for values, label, color in (
        (prepared[0], "highlum", HIGHLUM_COLOR),
        (prepared[1], "highdens", HIGHDENS_COLOR),
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

    axis.axvline(
        max_arcsec,
        color="tab:red",
        linestyle="--",
        label=f"{max_arcsec:g} arcsec limit",
    )
    axis.set_xscale("log")
    axis.set_xlim(lower, max_arcsec * 1.05)
    axis.set_xlabel("Coordinate-match separation [arcsec, log scale]")
    axis.set_ylabel("Normalized density")
    axis.set_title("Coordinate-match separation comparison")
    axis.legend()
    axis.grid(alpha=0.2)
    return figure
