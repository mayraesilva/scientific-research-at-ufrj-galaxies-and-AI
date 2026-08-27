"""Behavior tests for the nine comparison-focused Matplotlib figures."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.galaxy_analysis.plotting import (
    plot_catalogue_composition,
    plot_magnitude_radius_comparison,
    plot_model_disagreement_comparison,
    plot_morphology_brightness_size_comparison,
    plot_orientation_comparison,
    plot_probability_faintness_comparison,
    plot_robust_probability_comparison,
    plot_separation_comparison,
    plot_sky_footprint_comparison,
)
from src.galaxy_analysis.selection import robust_masks
from src.galaxy_analysis.statistics import composition_rows


SYNTHETIC_HIGHLUM = {
    "MAG_AUTO_R": np.array([18.0, 18.5, 19.0, 19.5, 20.0, 20.5]),
    "FLUX_RADIUS_R": np.array([3.0, 4.0, 5.0, 6.0, 5.5, 4.5]),
    "MP_LTG": np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95]),
    "MP_EdgeOn": np.array([0.01, 0.1, 0.2, 0.4, 0.6, 0.7]),
    "RA_2": np.array([10.0, 11.0, 12.0, 13.0, 12.5, 11.5]),
    "DEC_2": np.array([-3.0, -2.0, -1.0, 0.0, 0.5, -0.5]),
    "FLAG_LTG": np.array([4, 4, 4, 5, 5, 0]),
}
SYNTHETIC_HIGHDENS = {
    "MAG_AUTO_R": np.array([18.2, 18.7, 19.2, 19.7, 20.2, 20.7]),
    "FLUX_RADIUS_R": np.array([3.5, 4.5, 5.5, 6.5, 6.0, 5.0]),
    "MP_LTG": np.array([0.1, 0.3, 0.45, 0.65, 0.75, 0.9]),
    "MP_EdgeOn": np.array([0.02, 0.2, 0.3, 0.5, 0.7, 0.8]),
    "RA_2": np.array([10.5, 11.5, 12.5, 13.5, 13.0, 12.0]),
    "DEC_2": np.array([-2.5, -1.5, -0.5, 0.5, 1.0, 0.0]),
    "FLAG_LTG": np.array([4, 4, 5, 5, 5, 1]),
}


def test_catalogue_composition_has_count_and_fraction_panels():
    """Catch loss of absolute or normalized catalogue composition."""

    highlum = composition_rows("highlum", SYNTHETIC_HIGHLUM["FLAG_LTG"])
    highdens = composition_rows("highdens", SYNTHETIC_HIGHDENS["FLAG_LTG"])

    figure = plot_catalogue_composition(highlum, highdens)

    assert len(figure.axes) == 2
    assert figure.axes[0].get_ylabel() == "Galaxy count"
    assert figure.axes[1].get_ylabel() == "Fraction of matched catalogue"
    assert figure.axes[1].get_ylim() == pytest.approx((0.0, 1.0))
    assert figure.axes[0].get_legend_handles_labels()[1] == [
        "Robust ETG",
        "Robust LTG",
        "Other flags",
    ]
    plt.close(figure)


def test_robust_probability_comparison_normalizes_with_shared_limits():
    """Catch raw-count dominance or inconsistent probability axes."""

    figure = plot_robust_probability_comparison(
        highlum_values={
            "robust_etg": np.array([0.0, 0.1]),
            "robust_ltg": np.array([0.8, 1.0]),
        },
        highdens_values={
            "robust_etg": np.array([0.05, 0.2]),
            "robust_ltg": np.array([0.7, 0.95]),
        },
    )

    assert all(axis.get_xlim() == pytest.approx((0.0, 1.0)) for axis in figure.axes)
    assert all(axis.get_ylabel() == "Normalized density" for axis in figure.axes)
    plt.close(figure)


def test_model_disagreement_comparison_contains_both_catalogues():
    """Catch a panel that silently drops either selected catalogue."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        figure = plot_model_disagreement_comparison(
            {"robust_etg": np.array([0.01]), "robust_ltg": np.array([0.03])},
            {"robust_etg": np.array([0.02]), "robust_ltg": np.array([0.04])},
        )

    assert figure.axes[0].get_legend_handles_labels()[1] == [
        "highlum",
        "highdens",
    ]
    assert "standard deviation" in figure.axes[0].get_xlabel()
    plt.close(figure)


def test_magnitude_radius_comparison_uses_shared_inverted_axes():
    """Catch mismatched scales or the ordinary non-astronomical magnitude direction."""

    figure = plot_magnitude_radius_comparison(
        SYNTHETIC_HIGHLUM,
        SYNTHETIC_HIGHDENS,
    )
    left, right = figure.axes[:2]

    assert left.get_xlim() == pytest.approx(right.get_xlim())
    assert left.get_ylim() == pytest.approx(right.get_ylim())
    assert left.xaxis_inverted() and right.xaxis_inverted()
    assert left.get_ylabel() == "FLUX_RADIUS_R [pixel]"
    plt.close(figure)


def test_morphology_brightness_size_has_four_shared_class_panels():
    """Catch raw mixed-class scatter or incomparable class-panel scales."""

    figure = plot_morphology_brightness_size_comparison(
        SYNTHETIC_HIGHLUM,
        robust_masks(SYNTHETIC_HIGHLUM["FLAG_LTG"]),
        SYNTHETIC_HIGHDENS,
        robust_masks(SYNTHETIC_HIGHDENS["FLAG_LTG"]),
    )
    data_axes = figure.axes[:4]

    assert len(data_axes) == 4
    assert all(axis.xaxis_inverted() for axis in data_axes)
    assert len({axis.get_xlim() for axis in data_axes}) == 1
    assert len({axis.get_ylim() for axis in data_axes}) == 1
    assert "Robust ETG" in data_axes[0].get_title()
    assert "Robust LTG" in data_axes[1].get_title()
    plt.close(figure)


def test_probability_faintness_overlays_binned_median_and_iqr():
    """Catch density-only output that hides the measured binned trend."""

    figure = plot_probability_faintness_comparison(
        SYNTHETIC_HIGHLUM,
        SYNTHETIC_HIGHDENS,
        np.array([18.0, 19.0, 20.0, 21.0]),
    )

    for axis in figure.axes[:2]:
        assert len(axis.lines) == 1
        assert len(axis.collections) >= 2
        assert axis.get_xlabel().endswith("(fainter →)")
        assert axis.get_ylim() == pytest.approx((0.0, 1.0))
    plt.close(figure)


def test_orientation_comparison_uses_probability_domain_and_colorbar():
    """Catch an orientation plot with inconsistent domains or unexplained color."""

    figure = plot_orientation_comparison(SYNTHETIC_HIGHLUM, SYNTHETIC_HIGHDENS)
    left, right = figure.axes[:2]

    assert left.get_xlim() == pytest.approx((0.0, 1.0))
    assert right.get_xlim() == pytest.approx((0.0, 1.0))
    assert left.get_ylim() == pytest.approx((0.0, 1.0))
    assert figure.axes[-1].get_ylabel() == "Logarithmic count per hexagonal bin"
    plt.close(figure)


def test_sky_footprint_comparison_preserves_gaps_on_shared_limits():
    """Catch interpolation across holes or mismatched coordinate windows."""

    figure = plot_sky_footprint_comparison(SYNTHETIC_HIGHLUM, SYNTHETIC_HIGHDENS)
    left, right = figure.axes

    assert left.get_xlim() == pytest.approx(right.get_xlim())
    assert left.get_ylim() == pytest.approx(right.get_ylim())
    assert len(left.images) == 0 and len(right.images) == 0
    assert len(left.collections) == 1 and len(right.collections) == 1
    plt.close(figure)


def test_separation_comparison_marks_one_arcsecond_and_normalizes():
    """Catch a hidden tolerance or raw-count comparison between unequal files."""

    figure = plot_separation_comparison(
        np.array([0.1, 0.2, 0.5]),
        np.array([0.2, 0.4, 0.8]),
        max_arcsec=1.0,
    )
    axis = figure.axes[0]
    limit_lines = [
        line
        for line in axis.lines
        if np.allclose(line.get_xdata(), [1.0, 1.0])
    ]

    assert len(limit_lines) == 1
    assert axis.get_xscale() == "log"
    assert axis.get_xlabel() == "Coordinate-match separation [arcsec, log scale]"
    assert axis.get_ylabel() == "Normalized density"
    assert axis.get_legend_handles_labels()[1] == [
        "highlum",
        "highdens",
        "1 arcsec limit",
    ]
    plt.close(figure)
