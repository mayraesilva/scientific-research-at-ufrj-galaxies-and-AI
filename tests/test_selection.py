import numpy as np
import pytest

from src.galaxy_analysis.selection import (
    audit_filter,
    robust_masks,
    valid_value_mask,
    validate_separation,
)


def test_robust_masks_do_not_use_parity():
    masks = robust_masks(np.array([0, 1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(masks.robust_etg, [False, False, False, False, True, False])
    np.testing.assert_array_equal(masks.robust_ltg, [False, False, False, False, False, True])
    np.testing.assert_array_equal(masks.robust_any, [False, False, False, False, True, True])
    assert not np.any(masks.robust_etg & masks.robust_ltg)


def test_validate_separation_accepts_boundary():
    result = validate_separation(np.array([0.0, 0.5, 1.0]), 1.0)
    assert result.is_valid
    assert result.invalid_count == 0


def test_validate_separation_rejects_above_boundary_and_nonfinite():
    result = validate_separation(np.array([0.0, 1.01, np.nan]), 1.0)
    assert not result.is_valid
    assert result.invalid_count == 2


@pytest.mark.parametrize(
    ("domain", "values", "expected"),
    [
        ("ra_deg", [-0.1, 0.0, 359.9, 360.0, np.nan], [False, True, True, False, False]),
        ("dec_deg", [-90.1, -90.0, 90.0, 90.1, np.inf], [False, True, True, False, False]),
        ("positive", [-1.0, 0.0, 0.1, np.nan], [False, False, True, False]),
        ("probability", [-0.1, 0.0, 1.0, 1.1, np.inf], [False, True, True, False, False]),
        ("nonnegative", [-0.1, 0.0, 1.0, np.nan], [False, True, True, False]),
    ],
)
def test_valid_value_mask_enforces_domain(domain, values, expected):
    np.testing.assert_array_equal(valid_value_mask(np.array(values), domain), expected)


def test_valid_value_mask_rejects_unknown_domain():
    with pytest.raises(ValueError, match="unknown validation domain"):
        valid_value_mask(np.array([1.0]), "unknown")


def test_audit_filter_records_before_removed_and_after():
    row = audit_filter("quality", "finite probability", np.array([True, False, True]))
    assert row.n_before == 3
    assert row.n_removed == 1
    assert row.n_after == 2
    assert row.fraction_removed == pytest.approx(1 / 3)
