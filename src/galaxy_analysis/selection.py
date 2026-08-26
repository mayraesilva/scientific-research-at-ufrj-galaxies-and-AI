"""Validation rules and explicit robust morphology selections.

Centralizing these masks prevents different notebook sections from quietly
using different class definitions or validity domains.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MorphologyMasks:
    """Disjoint boolean masks for robust and excluded morphology rows."""

    robust_etg: np.ndarray
    robust_ltg: np.ndarray
    robust_any: np.ndarray
    non_robust: np.ndarray


@dataclass(frozen=True)
class FilterAuditRow:
    """One reproducible record of the effect of a validation/filter rule."""

    stage: str
    rule: str
    n_before: int
    n_removed: int
    n_after: int
    fraction_removed: float


@dataclass(frozen=True)
class SeparationValidation:
    """Summary of whether coordinate matches satisfy an angular limit."""

    is_valid: bool
    invalid_count: int
    maximum_valid: float | None


def robust_masks(flag_ltg: np.ndarray) -> MorphologyMasks:
    """Separate only flags 4 and 5 into robust ETG and LTG samples.

    Flags 0–3 encode lower-confidence catalogue outcomes.  Keeping them in a
    separate mask prevents the tempting even/odd shortcut from contaminating
    the primary robust comparison.
    """
    flags = np.asarray(flag_ltg)
    robust_etg = flags == 4
    robust_ltg = flags == 5
    robust_any = robust_etg | robust_ltg
    return MorphologyMasks(robust_etg, robust_ltg, robust_any, ~robust_any)


def valid_value_mask(values: np.ndarray, domain: str) -> np.ndarray:
    """Return finite values inside one named scientific validity domain.

    Named domains make every rule auditable in the notebook and keep NaN/Inf
    handling consistent across coordinates, radii, probabilities, and match
    separations.
    """
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
    """Validate nonnegative angular separations against a maximum in arcsec.

    The result reports both pass/fail state and counts so a bad match catalogue
    cannot proceed to normal-looking figures after silently dropping rows.
    """
    values = np.asarray(values, dtype=float)
    valid = valid_value_mask(values, "nonnegative") & (values <= maximum_arcsec)
    finite_valid = values[valid]
    maximum_valid = float(np.max(finite_valid)) if finite_valid.size else None
    invalid_count = int(values.size - np.count_nonzero(valid))
    return SeparationValidation(invalid_count == 0, invalid_count, maximum_valid)


def audit_filter(stage: str, rule: str, keep_mask: np.ndarray) -> FilterAuditRow:
    """Describe exactly how many rows a boolean keep-mask removes and why.

    Persisting these counts in ``filter_audit.csv`` makes provisional cuts and
    data-quality losses visible rather than implicit notebook state.
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    n_before = int(keep_mask.size)
    n_after = int(np.count_nonzero(keep_mask))
    n_removed = n_before - n_after
    fraction_removed = n_removed / n_before if n_before else 0.0
    return FilterAuditRow(stage, rule, n_before, n_removed, n_after, fraction_removed)
