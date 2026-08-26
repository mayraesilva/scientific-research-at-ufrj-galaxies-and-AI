import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.galaxy_analysis.plotting import (
    plot_edgeon_vs_ltg_probability,
    plot_flag_counts,
    plot_magnitude_radius_density,
    plot_magnitude_radius_scatter,
    plot_model_dispersion_by_class,
    plot_probability_by_class,
    plot_probability_vs_magnitude,
    plot_parameter_comparison,
    plot_robust_fraction_comparison,
    plot_separation_comparison,
    plot_sky_distribution,
)
from src.galaxy_analysis.selection import robust_masks
from src.galaxy_analysis.statistics import flag_count_rows


@pytest.fixture
def sample():
    return {
        "magnitude": np.array([18.0, 19.0, 20.0, 21.0, 22.0, np.nan]),
        "radius": np.array([3.0, 4.0, 7.0, 10.0, 60.0, 5.0]),
        "probability": np.array([0.02, 0.15, 0.8, 0.95, 0.6, np.nan]),
        "edgeon": np.array([0.01, 0.1, 0.3, 0.9, 0.7, np.nan]),
        "dispersion": np.array([0.01, 0.02, 0.1, 0.2, 0.15, np.nan]),
        "ra": np.array([10.0, 11.0, 12.0, 13.0, 14.0, np.nan]),
        "dec": np.array([-2.0, -1.0, 0.0, 1.0, 2.0, np.nan]),
        "flags": np.array([4, 4, 5, 5, 0, 1]),
    }


def assert_figure_labels(figure, xlabel, ylabel):
    assert isinstance(figure, Figure)
    assert figure.axes[0].get_xlabel() == xlabel
    assert figure.axes[0].get_ylabel() == ylabel
    plt.close(figure)


def test_magnitude_radius_scatter_labels_classes_and_inverts_magnitude(sample):
    magnitude = sample["magnitude"].copy()
    radius = sample["radius"].copy()
    masks = robust_masks(sample["flags"])
    figure = plot_magnitude_radius_scatter(magnitude, radius, masks, "highlum")
    axis = figure.axes[0]
    assert axis.get_xlabel() == "MAG_AUTO_R [mag] (brighter ←)"
    assert axis.get_ylabel() == "FLUX_RADIUS_R [pixel]"
    assert axis.xaxis_inverted()
    labels = axis.get_legend_handles_labels()[1]
    assert any("Robust ETG (N=" in label for label in labels)
    assert any("Robust LTG (N=" in label for label in labels)
    np.testing.assert_array_equal(magnitude, sample["magnitude"])
    np.testing.assert_array_equal(radius, sample["radius"])
    plt.close(figure)


def test_magnitude_radius_scatter_can_show_the_full_valid_catalog(sample):
    figure = plot_magnitude_radius_scatter(
        sample["magnitude"], sample["radius"], None, "highlum"
    )
    labels = figure.axes[0].get_legend_handles_labels()[1]
    assert labels == ["All valid rows (N=5)"]
    assert "all valid rows" in figure.axes[0].get_title()
    plt.close(figure)


def test_magnitude_radius_density_uses_physical_axis_labels(sample):
    figure = plot_magnitude_radius_density(
        sample["magnitude"], sample["radius"], "highlum", flux_radius_max=50
    )
    assert_figure_labels(
        figure, "MAG_AUTO_R [mag] (brighter ←)", "FLUX_RADIUS_R [pixel]"
    )


def test_probability_histogram_has_fixed_probability_axis_and_class_sizes(sample):
    figure = plot_probability_by_class(
        sample["probability"], robust_masks(sample["flags"]), "MP_LTG", "highlum"
    )
    axis = figure.axes[0]
    assert axis.get_xlim() == pytest.approx((0.0, 1.0))
    assert axis.get_xlabel() == "MP_LTG probability"
    assert axis.get_ylabel() == "Normalized density"
    labels = axis.get_legend_handles_labels()[1]
    assert all("N=" in label for label in labels)
    plt.close(figure)


def test_model_dispersion_plot_has_class_axis_and_units(sample):
    figure = plot_model_dispersion_by_class(
        sample["dispersion"], robust_masks(sample["flags"]), "highlum"
    )
    assert_figure_labels(
        figure, "Robust morphology class", "P1–P5 LTG sample standard deviation"
    )


def test_probability_vs_magnitude_has_fixed_probability_axis(sample):
    figure = plot_probability_vs_magnitude(
        sample["magnitude"], sample["probability"], "highlum"
    )
    axis = figure.axes[0]
    assert axis.get_ylim() == pytest.approx((0.0, 1.0))
    assert "N=5" in axis.get_title()
    assert_figure_labels(figure, "MAG_AUTO_R [mag] (fainter →)", "MP_LTG probability")


def test_edgeon_vs_ltg_probability_has_two_fixed_probability_axes(sample):
    figure = plot_edgeon_vs_ltg_probability(
        sample["probability"], sample["edgeon"], "highlum"
    )
    axis = figure.axes[0]
    assert axis.get_xlim() == pytest.approx((0.0, 1.0))
    assert axis.get_ylim() == pytest.approx((0.0, 1.0))
    assert "N=5" in axis.get_title()
    assert_figure_labels(figure, "MP_LTG probability", "MP_EdgeOn probability")


@pytest.mark.parametrize("with_masks", [False, True])
def test_sky_distribution_labels_coordinates_and_preserves_inputs(sample, with_masks):
    ra = sample["ra"].copy()
    dec = sample["dec"].copy()
    masks = robust_masks(sample["flags"]) if with_masks else None
    figure = plot_sky_distribution(ra, dec, masks, "highlum")
    assert_figure_labels(figure, "Right ascension RA_2 [deg]", "Declination DEC_2 [deg]")
    np.testing.assert_array_equal(ra, sample["ra"])
    np.testing.assert_array_equal(dec, sample["dec"])


def test_flag_counts_show_every_flag_and_percent(sample):
    flags = sample["flags"].copy()
    figure = plot_flag_counts(flags, "highlum")
    axis = figure.axes[0]
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [str(i) for i in range(6)]
    assert len(axis.texts) == 6
    assert all("%" in text.get_text() for text in axis.texts)
    np.testing.assert_array_equal(flags, sample["flags"])
    assert_figure_labels(figure, "FLAG_LTG value", "Galaxy count")


def test_robust_fraction_comparison_has_both_catalogues_and_probability_scale():
    highlum = flag_count_rows("highlum", np.array([4, 4, 4, 5, 0]))
    highdens = flag_count_rows("highdens", np.array([4, 5, 5, 5, 1]))
    figure = plot_robust_fraction_comparison(highlum, highdens)
    axis = figure.axes[0]
    assert axis.get_ylim() == pytest.approx((0.0, 1.0))
    assert axis.get_legend_handles_labels()[1] == ["highlum", "highdens"]
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "Robust ETG (flag 4)",
        "Robust LTG (flag 5)",
    ]
    plt.close(figure)


def test_parameter_comparison_uses_shared_limits_and_normalized_density():
    figure = plot_parameter_comparison(
        "MP_LTG",
        {"robust_etg": np.array([0.0, 0.1]), "robust_ltg": np.array([0.8, 1.0])},
        {"robust_etg": np.array([0.05, 0.2]), "robust_ltg": np.array([0.7, 0.95])},
    )
    data_axes = figure.axes[:2]
    assert all(axis.get_xlim() == pytest.approx((0.0, 1.0)) for axis in data_axes)
    assert all(axis.get_ylabel() == "Normalized density" for axis in data_axes)
    assert all(axis.get_legend_handles_labels()[1] == ["highlum", "highdens"] for axis in data_axes)
    plt.close(figure)


def test_separation_comparison_marks_one_arcsecond_and_normalizes():
    figure = plot_separation_comparison(
        np.array([0.1, 0.2, 0.5]), np.array([0.2, 0.4, 0.8]), max_arcsec=1.0
    )
    axis = figure.axes[0]
    reference_lines = [line for line in axis.lines if np.allclose(line.get_xdata(), [1.0, 1.0])]
    assert len(reference_lines) == 1
    assert axis.get_xscale() == "log"
    assert axis.get_xlabel() == "Coordinate-match separation [arcsec, log scale]"
    assert axis.get_ylabel() == "Normalized density"
    assert axis.get_legend_handles_labels()[1] == ["highlum", "highdens", "1 arcsec limit"]
    plt.close(figure)
