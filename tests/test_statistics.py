import numpy as np
import pytest

from src.galaxy_analysis.statistics import (
    binned_quantiles,
    compare_catalog_summaries,
    composition_rows,
    describe_values,
    eligible_binned_bins,
    flag_count_rows,
    model_dispersion,
    threshold_rows,
    wilson_interval,
)


def test_composition_rows_group_only_flags_4_and_5_as_robust():
    """Catch accidental parity selection or loss of excluded flags."""

    rows = composition_rows("sample", np.array([0, 1, 2, 3, 4, 4, 5]))
    by_class = {row.class_name: row for row in rows}

    assert by_class["robust_etg"].count == 2
    assert by_class["robust_ltg"].count == 1
    assert by_class["other_flags"].count == 4
    assert sum(row.fraction_of_total for row in rows) == pytest.approx(1.0)


def test_composition_rows_include_counting_precision():
    """Catch grouped rows that omit their Wilson counting intervals."""

    rows = composition_rows("sample", np.array([4, 4, 5, 0]))

    assert all(row.wilson_low_95 is not None for row in rows)
    assert all(row.wilson_high_95 is not None for row in rows)


def test_binned_quantiles_return_counts_median_and_iqr():
    """Catch incorrect bin membership or quartile ordering."""

    result = binned_quantiles(
        x=np.array([18.1, 18.2, 19.1, 19.2]),
        y=np.array([0.0, 0.2, 0.8, 1.0]),
        bin_edges=np.array([18.0, 19.0, 20.0]),
    )

    np.testing.assert_array_equal(result.count, [2, 2])
    np.testing.assert_allclose(result.median, [0.1, 0.9])
    np.testing.assert_allclose(result.p25, [0.05, 0.85])
    np.testing.assert_allclose(result.p75, [0.15, 0.95])


def test_binned_quantiles_mark_empty_bins_with_nan():
    """Catch fabricated zero-valued statistics for empty magnitude bins."""

    result = binned_quantiles(
        x=np.array([18.2]),
        y=np.array([0.4]),
        bin_edges=np.array([18.0, 19.0, 20.0]),
    )

    assert result.count.tolist() == [1, 0]
    assert np.isnan(result.median[1])


def test_binned_quantiles_include_the_final_right_edge():
    """Catch loss of the faintest value when it equals the pooled maximum."""

    result = binned_quantiles(
        x=np.array([18.0, 19.0, 20.0]),
        y=np.array([0.1, 0.5, 0.9]),
        bin_edges=np.array([18.0, 19.0, 20.0]),
    )

    assert result.count.tolist() == [1, 2]
    assert result.median[1] == pytest.approx(0.7)


def test_binned_quantiles_reject_nonincreasing_edges():
    """Catch ambiguous or overlapping bin definitions."""

    with pytest.raises(ValueError, match="increasing"):
        binned_quantiles(
            x=np.array([18.2]),
            y=np.array([0.4]),
            bin_edges=np.array([18.0, 20.0, 19.0]),
        )


def test_eligible_binned_bins_enforces_minimum_occupancy():
    """Catch sparse bins being treated as stable descriptive endpoints."""

    result = binned_quantiles(
        x=np.array([18.1, 19.1, 19.2]),
        y=np.array([0.9, 0.2, 0.4]),
        bin_edges=np.array([18.0, 19.0, 20.0]),
    )

    np.testing.assert_array_equal(eligible_binned_bins(result, 2), [False, True])
    with pytest.raises(ValueError, match="at least 1"):
        eligible_binned_bins(result, 0)


def test_describe_values_uses_sample_standard_deviation():
    row = describe_values("test", "robust_etg", "x", np.array([1.0, 2.0, 3.0]))
    assert row.n_valid == 3
    assert row.mean == pytest.approx(2.0)
    assert row.median == pytest.approx(2.0)
    assert row.std_ddof1 == pytest.approx(1.0)
    assert row.p25 == pytest.approx(1.5)
    assert row.p75 == pytest.approx(2.5)
    assert row.iqr == pytest.approx(1.0)


def test_describe_values_excludes_nonfinite_values():
    row = describe_values("test", "all", "x", np.array([1.0, np.nan, np.inf, 3.0]))
    assert row.n_total == 4
    assert row.n_valid == 2
    assert row.n_missing == 2
    assert row.mean == pytest.approx(2.0)


def test_describe_values_omits_std_for_single_value():
    row = describe_values("test", "robust_ltg", "x", np.array([2.0]))
    assert row.std_ddof1 is None


def test_model_dispersion_uses_five_models():
    values = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
    result = model_dispersion(values)
    assert result["median"][0] == pytest.approx(0.5)
    assert result["range"][0] == pytest.approx(1.0)
    assert result["std_ddof1"][0] == pytest.approx(np.std(values[0], ddof=1))


def test_model_dispersion_requires_five_columns():
    with pytest.raises(ValueError, match=r"shape \(rows, 5\)"):
        model_dispersion(np.ones((3, 4)))


def test_wilson_interval_bounds_zero_and_full_counts():
    low_zero, high_zero = wilson_interval(0, 10)
    low_full, high_full = wilson_interval(10, 10)
    assert low_zero == pytest.approx(0.0)
    assert 0 < high_zero < 1
    assert 0 < low_full < 1
    assert high_full == pytest.approx(1.0)


def test_wilson_interval_has_no_value_for_empty_population():
    assert wilson_interval(0, 0) == (None, None)


def test_flag_count_rows_preserve_all_flags_and_labels():
    rows = flag_count_rows("test", np.array([0, 4, 4, 5]))
    by_flag = {row.flag_ltg: row for row in rows}
    assert by_flag[4].count == 2
    assert by_flag[4].class_label == "robust_etg"
    assert by_flag[5].class_label == "robust_ltg"
    assert by_flag[0].class_label == "non_robust_etg"


def test_threshold_rows_report_hand_derived_confusion_counts():
    probabilities = np.array([0.1, 0.8, 0.6, 0.4])
    flags = np.array([4, 5, 4, 5])
    row = threshold_rows("test", probabilities, flags, (0.5,))[0]
    assert row.true_positive == 1
    assert row.true_negative == 1
    assert row.false_positive == 1
    assert row.false_negative == 1
    assert row.agreement_with_flag5 == pytest.approx(0.5)


def test_compare_catalog_summaries_uses_highdens_minus_highlum():
    highlum = [
        describe_values("highlum", "all_valid", "MP_LTG", np.arange(10.0)),
        describe_values("highlum", "robust_etg", "MP_LTG", np.array([0.1, 0.2, 0.3])),
    ]
    highdens = [
        describe_values("highdens", "all_valid", "MP_LTG", np.arange(20.0)),
        describe_values("highdens", "robust_etg", "MP_LTG", np.array([0.3, 0.4, 0.5, 0.6])),
    ]
    row = compare_catalog_summaries(highlum, highdens)[0]
    assert row.variable == "MP_LTG"
    assert row.class_name == "robust_etg"
    assert row.highlum_median == pytest.approx(0.2)
    assert row.highdens_median == pytest.approx(0.45)
    assert row.median_difference == pytest.approx(0.25)
    assert row.highlum_fraction == pytest.approx(3 / 10)
    assert row.highdens_fraction == pytest.approx(4 / 20)
    assert row.fraction_difference == pytest.approx(4 / 20 - 3 / 10)
