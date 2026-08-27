"""Tests for result-specific, scientifically cautious notebook explanations."""

import numpy as np

from src.galaxy_analysis.reporting import (
    brightness_size_class_interpretation,
    composition_interpretation,
    dispersion_interpretation,
    faintness_interpretation,
    magnitude_radius_interpretation,
    orientation_interpretation,
    probability_interpretation,
    radius_cut_interpretation,
    separation_interpretation,
    sky_interpretation,
)
from src.galaxy_analysis.statistics import (
    BinnedSummary,
    composition_rows,
    describe_values,
)


def test_composition_interpretation_reports_counts_and_signed_difference():
    """Catch prose that hides class imbalance or reverses catalogue subtraction."""

    highlum = composition_rows("highlum", np.array([4, 4, 4, 5, 0]))
    highdens = composition_rows("highdens", np.array([4, 5, 5, 5, 1]))

    text = composition_interpretation(highlum, highdens)

    assert "3 robust ETG classifications" in text
    assert "3 robust LTG classifications" in text
    assert "highdens - highlum" in text
    assert "+40.000%" in text
    assert "caused" not in text.lower()


def test_probability_interpretation_uses_classification_language():
    """Catch model outputs described as true galaxies or independent accuracy."""

    rows = [
        describe_values("highlum", "robust_etg", "MP_LTG", np.array([0.0, 0.1])),
        describe_values("highlum", "robust_ltg", "MP_LTG", np.array([0.8, 1.0])),
        describe_values("highdens", "robust_etg", "MP_LTG", np.array([0.1, 0.2])),
        describe_values("highdens", "robust_ltg", "MP_LTG", np.array([0.7, 0.9])),
    ]

    text = probability_interpretation(rows)

    assert "robust ETG classifications" in text
    assert "robust LTG classifications" in text
    assert "median MP_LTG=0.050" in text
    assert "true galaxy" not in text.lower()
    assert "independently measuring classification accuracy" in text


def test_dispersion_interpretation_names_largest_measured_median():
    """Catch the wrong largest class or a claim of total uncertainty."""

    rows = [
        describe_values("highlum", "robust_etg", "P1_P5_LTG_STD", np.array([0.01])),
        describe_values("highlum", "robust_ltg", "P1_P5_LTG_STD", np.array([0.03])),
        describe_values("highdens", "robust_etg", "P1_P5_LTG_STD", np.array([0.02])),
        describe_values("highdens", "robust_ltg", "P1_P5_LTG_STD", np.array([0.04])),
    ]

    text = dispersion_interpretation(rows)

    assert "highdens robust LTG classifications" in text
    assert "0.040" in text
    assert "not the full uncertainty" in text


def test_magnitude_radius_interpretation_uses_observed_pixel_units():
    """Catch apparent measurements incorrectly described as physical quantities."""

    rows = [
        describe_values("highlum", "all_valid", "MAG_AUTO_R", np.array([18.0, 20.0])),
        describe_values("highlum", "all_valid", "FLUX_RADIUS_R", np.array([3.0, 5.0])),
        describe_values("highdens", "all_valid", "MAG_AUTO_R", np.array([17.0, 21.0])),
        describe_values("highdens", "all_valid", "FLUX_RADIUS_R", np.array([4.0, 8.0])),
    ]

    text = magnitude_radius_interpretation(rows)

    assert "apparent magnitudes 18.00 to 20.00" in text
    assert "median half-light radius 4.00 pixels" in text
    assert "not physical size" in text


def test_brightness_size_class_interpretation_compares_robust_medians():
    """Catch class comparison prose that ignores the measured class summaries."""

    rows = [
        describe_values("highlum", "robust_etg", "FLUX_RADIUS_R", np.array([3.0, 5.0])),
        describe_values("highlum", "robust_ltg", "FLUX_RADIUS_R", np.array([5.0, 7.0])),
        describe_values(
            "highdens", "robust_etg", "FLUX_RADIUS_R", np.array([4.0, 6.0])
        ),
        describe_values(
            "highdens", "robust_ltg", "FLUX_RADIUS_R", np.array([6.0, 8.0])
        ),
    ]

    text = brightness_size_class_interpretation(rows)

    assert "highlum" in text and "highdens" in text
    assert "2.00 pixels" in text
    assert "shared imaging measurements" in text


def test_faintness_interpretation_describes_measured_endpoint_change():
    """Catch trend prose that omits endpoint values or claims an accuracy change."""

    summary = BinnedSummary(
        bin_edges=np.array([18.0, 19.0, 20.0]),
        bin_centers=np.array([18.5, 19.5]),
        count=np.array([20, 30]),
        median=np.array([0.2, 0.5]),
        p25=np.array([0.1, 0.3]),
        p75=np.array([0.4, 0.7]),
    )

    text = faintness_interpretation("highlum", summary)

    assert "0.200" in text and "0.500" in text
    assert "independent truth labels" in text


def test_orientation_interpretation_keeps_orientation_distinct():
    """Catch MP_EdgeOn presented as another morphology class."""

    rows = [
        describe_values("highlum", "robust_etg", "MP_EdgeOn", np.array([0.1])),
        describe_values("highlum", "robust_ltg", "MP_EdgeOn", np.array([0.4])),
        describe_values("highdens", "robust_etg", "MP_EdgeOn", np.array([0.2])),
        describe_values("highdens", "robust_ltg", "MP_EdgeOn", np.array([0.5])),
    ]

    text = orientation_interpretation(rows)

    assert "ETG median=0.100" in text
    assert "LTG median=0.500" in text
    assert "orientation output, not a third morphology class" in text


def test_sky_interpretation_reports_ranges_and_mask_caution():
    """Catch footprint holes interpreted as physical underdensities."""

    text = sky_interpretation(
        highlum_ra=np.array([10.0, 12.0]),
        highlum_dec=np.array([-2.0, 1.0]),
        highdens_ra=np.array([9.0, 13.0]),
        highdens_dec=np.array([-3.0, 2.0]),
    )

    assert "RA 10.00-12.00 deg" in text
    assert "bright-star masks" in text
    assert "must not be interpreted as physical galaxy underdensities" in text


def test_separation_interpretation_reports_percentiles_and_angular_limit():
    """Catch physical-distance language or omission of match-quality tails."""

    text = separation_interpretation(
        highlum=np.array([0.1, 0.2, 0.3]),
        highdens=np.array([0.2, 0.4, 0.6]),
        maximum_arcsec=1.0,
    )

    assert "median (95th percentile)" in text
    assert "0.2 (0.29) arcsec for highlum" in text
    assert "1 arcsec angular tolerance" in text
    assert "not a physical distance" in text


def test_radius_cut_interpretation_reports_when_nothing_is_removed():
    """Catch a no-op provisional cut presented as a changed scientific sample."""

    text = radius_cut_interpretation(
        highlum_radius=np.array([3.0, 5.0]),
        highdens_radius=np.array([4.0, 60.0]),
        maximum_pixels=50.0,
    )

    assert "removes 0 highlum rows and 1 highdens row" in text
    assert "not applied to the primary statistics" in text
