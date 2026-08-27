"""Descriptive and classification statistics for morphology catalogues.

Functions here return immutable rows or NumPy arrays rather than printing or
writing files.  Keeping calculation separate from notebook presentation makes
the scientific definitions unit-testable and reusable for both catalogues.
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class SummaryRow:
    """Complete finite-value summary for one variable and sample class."""

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
    """Count, full-catalogue fraction, and Wilson interval for one flag."""

    catalog: str
    flag_ltg: int
    class_label: str
    count: int
    fraction_of_total: float
    wilson_low_95: float | None
    wilson_high_95: float | None


@dataclass(frozen=True)
class ThresholdRow:
    """Sensitivity comparison between one probability cut and flag 5."""

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


@dataclass(frozen=True)
class ComparisonRow:
    """Paired catalogue result with differences defined highdens − highlum."""

    variable: str
    class_name: str
    highlum_median: float | None
    highdens_median: float | None
    median_difference: float | None
    highlum_fraction: float
    highdens_fraction: float
    fraction_difference: float


@dataclass(frozen=True)
class CompositionRow:
    """One mutually exclusive morphology group in a matched catalogue."""

    catalog: str
    class_name: str
    count: int
    fraction_of_total: float
    wilson_low_95: float | None
    wilson_high_95: float | None


@dataclass(frozen=True)
class BinnedSummary:
    """Counts and y-distribution quartiles measured in fixed x intervals."""

    bin_edges: np.ndarray
    bin_centers: np.ndarray
    count: np.ndarray
    median: np.ndarray
    p25: np.ndarray
    p75: np.ndarray


def wilson_interval(count: int, total: int) -> tuple[float | None, float | None]:
    """Return the two-sided 95% Wilson interval for a binomial proportion.

    Wilson bounds behave better than ``p ± 1.96*sqrt(p(1-p)/n)`` near zero and
    one—the exact situation for rare robust LTGs.  ``None`` for an empty
    population avoids inventing numerical precision where no estimate exists.
    """
    if total == 0:
        return None, None
    if count < 0 or total < 0 or count > total:
        raise ValueError("count must satisfy 0 <= count <= total")
    z = 1.959963984540054
    proportion = count / total
    # This is the closed-form Wilson score interval, so SciPy is unnecessary.
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def composition_rows(catalog: str, flags: np.ndarray) -> list[CompositionRow]:
    """Group robust ETGs, robust LTGs, and all flags excluded from analysis.

    The groups are deliberately mutually exclusive.  Keeping flags 0-3 in one
    explicit group shows how much of each matched catalogue is not used in the
    primary robust comparison without relabelling those objects by parity.
    """

    flags = np.asarray(flags)
    masks = {
        "robust_etg": flags == 4,
        "robust_ltg": flags == 5,
        "other_flags": ~np.isin(flags, (4, 5)),
    }
    total = int(flags.size)
    rows = []

    for class_name, mask in masks.items():
        count = int(np.count_nonzero(mask))
        low, high = wilson_interval(count, total)
        rows.append(
            CompositionRow(
                catalog=catalog,
                class_name=class_name,
                count=count,
                fraction_of_total=count / total if total else 0.0,
                wilson_low_95=low,
                wilson_high_95=high,
            )
        )

    return rows


def binned_quantiles(
    x: np.ndarray,
    y: np.ndarray,
    bin_edges: np.ndarray,
) -> BinnedSummary:
    """Summarize finite y values inside fixed, increasing x intervals.

    Empty bins remain ``NaN`` instead of receiving invented zero-valued
    statistics.  This makes gaps explicit when the binned median and
    interquartile band are drawn over the probability-density figures.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)

    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be one-dimensional and increasing")

    valid = np.isfinite(x) & np.isfinite(y)
    valid_x = x[valid]
    bin_count = edges.size - 1
    bin_index = np.digitize(valid_x, edges, right=False) - 1
    # NumPy assigns the final right edge beyond the last half-open interval.
    # Include that one boundary so a pooled maximum is not silently discarded.
    bin_index[valid_x == edges[-1]] = bin_count - 1
    values = y[valid]
    count = np.zeros(bin_count, dtype=int)
    median = np.full(bin_count, np.nan)
    p25 = np.full(bin_count, np.nan)
    p75 = np.full(bin_count, np.nan)

    for index in range(bin_count):
        selected = values[bin_index == index]
        count[index] = selected.size
        if selected.size:
            p25[index], median[index], p75[index] = np.percentile(
                selected,
                [25, 50, 75],
            )

    return BinnedSummary(
        bin_edges=edges,
        bin_centers=(edges[:-1] + edges[1:]) / 2,
        count=count,
        median=median,
        p25=p25,
        p75=p75,
    )


def eligible_binned_bins(
    summary: BinnedSummary,
    minimum_count: int,
) -> np.ndarray:
    """Select bins with enough rows for a stable descriptive comparison."""

    if minimum_count < 1:
        raise ValueError("minimum_count must be at least 1")
    return np.asarray(summary.count) >= minimum_count


def describe_values(
    catalog: str,
    class_name: str,
    variable: str,
    values: np.ndarray,
) -> SummaryRow:
    """Describe finite values with sample dispersion and fixed percentiles.

    Nonfinite values are counted as missing rather than silently influencing
    results.  Sample standard deviation uses ``ddof=1`` and is deliberately
    absent when fewer than two valid observations exist.
    """
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    n_total = int(values.size)
    n_valid = int(valid.size)
    if n_valid == 0:
        # Preserve the requested row in output tables even when every value is
        # missing; absent statistics are represented by None, not string NaN.
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
    """Summarize the five model probabilities independently for each object.

    The spread is an inter-model disagreement diagnostic, not a complete
    uncertainty estimate: the networks may share training data and biases.
    """
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
    """Count every observed flag and attach its documented confidence label.

    All flags remain visible even though only 4 and 5 enter the robust sample;
    this makes the scale of exclusions explicit in tables and bar charts.
    """
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
    """Compare probability thresholds with flag 5 as an operational reference.

    The confusion-style cells measure internal catalogue agreement only.  They
    are retained separately so readers do not mistake one aggregate agreement
    percentage for validation against independent physical truth.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    flags = np.asarray(flags)
    valid = np.isfinite(probabilities) & np.isfinite(flags)
    probabilities = probabilities[valid]
    flags = flags[valid]
    reference = flags == 5
    total = int(probabilities.size)
    rows = []
    for threshold in thresholds:
        # A positive prediction means an LTG probability at or above the
        # sensitivity threshold; this never replaces the primary flag rule.
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


def compare_catalog_summaries(
    highlum_rows: list[SummaryRow],
    highdens_rows: list[SummaryRow],
) -> list[ComparisonRow]:
    """Compare class summaries with every difference defined highdens − highlum.

    A single sign convention prevents ambiguous tables.  Class fractions use
    the corresponding ``all_valid`` row as denominator, while medians remain
    variable- and class-specific.
    """
    highdens_by_key = {(row.variable, row.class_name): row for row in highdens_rows}
    highlum_totals = {
        row.variable: row.n_total for row in highlum_rows if row.class_name == "all_valid"
    }
    highdens_totals = {
        row.variable: row.n_total for row in highdens_rows if row.class_name == "all_valid"
    }
    comparisons = []
    for highlum in highlum_rows:
        # ``all_valid`` supplies denominators; it is not itself a class contrast.
        if highlum.class_name == "all_valid":
            continue
        key = (highlum.variable, highlum.class_name)
        if key not in highdens_by_key:
            raise ValueError(f"highdens summary is missing {key}")
        if highlum.variable not in highlum_totals or highlum.variable not in highdens_totals:
            raise ValueError(f"all_valid denominator is missing for {highlum.variable}")
        highdens = highdens_by_key[key]
        highlum_fraction = highlum.n_total / highlum_totals[highlum.variable]
        highdens_fraction = highdens.n_total / highdens_totals[highlum.variable]
        median_difference = (
            highdens.median - highlum.median
            if highdens.median is not None and highlum.median is not None
            else None
        )
        comparisons.append(
            ComparisonRow(
                variable=highlum.variable,
                class_name=highlum.class_name,
                highlum_median=highlum.median,
                highdens_median=highdens.median,
                median_difference=median_difference,
                highlum_fraction=highlum_fraction,
                highdens_fraction=highdens_fraction,
                fraction_difference=highdens_fraction - highlum_fraction,
            )
        )
    return comparisons
