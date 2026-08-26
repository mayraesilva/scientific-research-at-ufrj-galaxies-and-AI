"""Descriptive and classification statistics for morphology catalogues."""

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class SummaryRow:
    catalog: str
    class_name: str
    variable: str
    n_total: int
    n_valid: int
    n_missing: int
    mean: float | None
    median: float | None
    std_ddof1: float | None
    minimum: float | None
    p05: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p95: float | None
    maximum: float | None
    iqr: float | None


@dataclass(frozen=True)
class FlagCountRow:
    catalog: str
    flag_ltg: int
    class_label: str
    count: int
    fraction_of_total: float
    wilson_low_95: float | None
    wilson_high_95: float | None


@dataclass(frozen=True)
class ThresholdRow:
    catalog: str
    threshold: float
    n_valid: int
    n_above: int
    fraction_above: float
    n_below: int
    fraction_below: float
    n_flag5: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    agreement_with_flag5: float


def wilson_interval(count: int, total: int) -> tuple[float | None, float | None]:
    """Return the two-sided 95% Wilson interval for a binomial proportion."""
    if total == 0:
        return None, None
    if count < 0 or total < 0 or count > total:
        raise ValueError("count must satisfy 0 <= count <= total")
    z = 1.959963984540054
    proportion = count / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def describe_values(
    catalog: str,
    class_name: str,
    variable: str,
    values: np.ndarray,
) -> SummaryRow:
    """Describe finite values with sample dispersion and fixed percentiles."""
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    n_total = int(values.size)
    n_valid = int(valid.size)
    if n_valid == 0:
        return SummaryRow(catalog, class_name, variable, n_total, 0, n_total, *([None] * 11))
    p05, p25, p50, p75, p95 = np.percentile(valid, [5, 25, 50, 75, 95])
    return SummaryRow(
        catalog=catalog,
        class_name=class_name,
        variable=variable,
        n_total=n_total,
        n_valid=n_valid,
        n_missing=n_total - n_valid,
        mean=float(np.mean(valid)),
        median=float(np.median(valid)),
        std_ddof1=float(np.std(valid, ddof=1)) if n_valid >= 2 else None,
        minimum=float(np.min(valid)),
        p05=float(p05),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p95=float(p95),
        maximum=float(np.max(valid)),
        iqr=float(p75 - p25),
    )


def model_dispersion(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize five model probabilities independently for each object."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError("probabilities must have shape (rows, 5)")
    return {
        "mean": np.mean(probabilities, axis=1),
        "median": np.median(probabilities, axis=1),
        "std_ddof1": np.std(probabilities, axis=1, ddof=1),
        "range": np.ptp(probabilities, axis=1),
    }


def flag_count_rows(catalog: str, flags: np.ndarray) -> list[FlagCountRow]:
    """Count every observed morphology flag with a documented class label."""
    flags = np.asarray(flags)
    values, counts = np.unique(flags, return_counts=True)
    labels = {
        0: "non_robust_etg",
        1: "non_robust_ltg",
        2: "non_robust_etg",
        3: "non_robust_ltg",
        4: "robust_etg",
        5: "robust_ltg",
    }
    total = int(flags.size)
    rows = []
    for value, count in zip(values, counts, strict=True):
        value_int = int(value)
        count_int = int(count)
        low, high = wilson_interval(count_int, total)
        rows.append(
            FlagCountRow(
                catalog,
                value_int,
                labels.get(value_int, "unknown"),
                count_int,
                count_int / total if total else 0.0,
                low,
                high,
            )
        )
    return rows


def threshold_rows(
    catalog: str,
    probabilities: np.ndarray,
    flags: np.ndarray,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.8),
) -> list[ThresholdRow]:
    """Compare probability thresholds with flag 5 as an operational reference."""
    probabilities = np.asarray(probabilities, dtype=float)
    flags = np.asarray(flags)
    valid = np.isfinite(probabilities) & np.isfinite(flags)
    probabilities = probabilities[valid]
    flags = flags[valid]
    reference = flags == 5
    total = int(probabilities.size)
    rows = []
    for threshold in thresholds:
        prediction = probabilities >= threshold
        tp = int(np.count_nonzero(prediction & reference))
        tn = int(np.count_nonzero(~prediction & ~reference))
        fp = int(np.count_nonzero(prediction & ~reference))
        fn = int(np.count_nonzero(~prediction & reference))
        n_above = int(np.count_nonzero(prediction))
        rows.append(
            ThresholdRow(
                catalog=catalog,
                threshold=float(threshold),
                n_valid=total,
                n_above=n_above,
                fraction_above=n_above / total if total else 0.0,
                n_below=total - n_above,
                fraction_below=(total - n_above) / total if total else 0.0,
                n_flag5=int(np.count_nonzero(reference)),
                true_positive=tp,
                true_negative=tn,
                false_positive=fp,
                false_negative=fn,
                agreement_with_flag5=(tp + tn) / total if total else 0.0,
            )
        )
    return rows
