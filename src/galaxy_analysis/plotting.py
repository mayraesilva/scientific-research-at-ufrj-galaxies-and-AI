"""Pure Matplotlib figure constructors for galaxy morphology analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .selection import MorphologyMasks


ETG_COLOR = "tab:red"
LTG_COLOR = "tab:blue"


def _finite_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(first, dtype=float)) & np.isfinite(
        np.asarray(second, dtype=float)
    )


def _finish(figure: Figure) -> Figure:
    figure.tight_layout()
    return figure


def plot_magnitude_radius_scatter(
    magnitude: np.ndarray,
    radius: np.ndarray,
    masks: MorphologyMasks | None,
    catalog_alias: str,
    flux_radius_max: float | None = None,
) -> Figure:
    """Plot robust classes in magnitude–angular-radius space."""
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
                rasterized=True,
                label=f"{label} (N={np.count_nonzero(selected):,})",
            )
    suffix = "" if flux_radius_max is None else f"; FLUX_RADIUS_R < {flux_radius_max:g}"
    sample_name = "all valid rows" if masks is None else "robust morphology"
    axis.set_title(f"{catalog_alias}: {sample_name} magnitude–radius{suffix}")
    axis.set_xlabel("MAG_AUTO_R [mag] (brighter ←)")
    axis.set_ylabel("FLUX_RADIUS_R [pixel]")
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
    """Plot the full valid magnitude–radius density as logarithmic hexagons."""
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
    """Compare normalized probability distributions for robust ETGs and LTGs."""
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
    """Show per-object P1–P5 LTG dispersion with median and IQR by class."""
    dispersion = np.asarray(dispersion, dtype=float)
    groups = []
    for class_mask in (masks.robust_etg, masks.robust_ltg):
        selected = np.isfinite(dispersion) & np.asarray(class_mask, dtype=bool)
        groups.append(dispersion[selected])

    figure, axis = plt.subplots(figsize=(7, 5))
    boxes = axis.boxplot(
        groups,
        tick_labels=[f"Robust ETG\nN={groups[0].size:,}", f"Robust LTG\nN={groups[1].size:,}"],
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
    """Plot LTG probability density against apparent magnitude."""
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
    """Plot joint density of LTG and edge-on probability estimates."""
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
    """Plot catalogue coordinates without interpolating across survey gaps."""
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
    """Plot counts and percentages for all FLAG_LTG values 0 through 5."""
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
