import numpy as np
import pytest

from src.galaxy_analysis.statistics import (
    describe_values,
    flag_count_rows,
    model_dispersion,
    threshold_rows,
    wilson_interval,
)


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
