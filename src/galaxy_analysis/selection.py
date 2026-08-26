"""Validation rules and robust morphology selections."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MorphologyMasks:
    robust_etg: np.ndarray
    robust_ltg: np.ndarray
    robust_any: np.ndarray
    non_robust: np.ndarray


@dataclass(frozen=True)
class FilterAuditRow:
    stage: str
    rule: str
    n_before: int
    n_removed: int
    n_after: int
    fraction_removed: float


@dataclass(frozen=True)
class SeparationValidation:
    is_valid: bool
    invalid_count: int
    maximum_valid: float | None


def robust_masks(flag_ltg: np.ndarray) -> MorphologyMasks:
    """Separate only flags 4 and 5 into the robust ETG/LTG samples."""
    flags = np.asarray(flag_ltg)
    robust_etg = flags == 4
    robust_ltg = flags == 5
    robust_any = robust_etg | robust_ltg
    return MorphologyMasks(robust_etg, robust_ltg, robust_any, ~robust_any)


def valid_value_mask(values: np.ndarray, domain: str) -> np.ndarray:
    """Return finite values inside one named scientific domain."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    rules = {
        "ra_deg": finite & (values >= 0) & (values < 360),
        "dec_deg": finite & (values >= -90) & (values <= 90),
        "positive": finite & (values > 0),
        "probability": finite & (values >= 0) & (values <= 1),
        "nonnegative": finite & (values >= 0),
    }
    try:
        return rules[domain]
    except KeyError as exc:
        raise ValueError(f"unknown validation domain: {domain}") from exc


def validate_separation(values: np.ndarray, maximum_arcsec: float) -> SeparationValidation:
    """Validate a nonnegative angular match separation threshold."""
    values = np.asarray(values, dtype=float)
    valid = valid_value_mask(values, "nonnegative") & (values <= maximum_arcsec)
    finite_valid = values[valid]
    maximum_valid = float(np.max(finite_valid)) if finite_valid.size else None
    invalid_count = int(values.size - np.count_nonzero(valid))
    return SeparationValidation(invalid_count == 0, invalid_count, maximum_valid)


def audit_filter(stage: str, rule: str, keep_mask: np.ndarray) -> FilterAuditRow:
    """Describe exactly how many rows a boolean filter removes."""
    keep_mask = np.asarray(keep_mask, dtype=bool)
    n_before = int(keep_mask.size)
    n_after = int(np.count_nonzero(keep_mask))
    n_removed = n_before - n_after
    fraction_removed = n_removed / n_before if n_before else 0.0
    return FilterAuditRow(stage, rule, n_before, n_removed, n_after, fraction_removed)
