"""Turn calculated morphology summaries into cautious research explanations.

The functions in this module do not inspect figures or invent conclusions.
They receive the same numerical summaries used by the plots and describe only
those measurements.  This keeps notebook prose natural while ensuring that a
catalogue update cannot leave hard-coded numbers behind.
"""

import numpy as np

from .statistics import (
    BinnedSummary,
    CompositionRow,
    SummaryRow,
    eligible_binned_bins,
)


CLASS_LABELS = {
    "robust_etg": "robust ETG",
    "robust_ltg": "robust LTG",
}


def _composition_lookup(
    rows: list[CompositionRow],
    class_name: str,
) -> CompositionRow:
    """Return one named composition group or fail on incomplete input."""

    matches = [row for row in rows if row.class_name == class_name]
    if len(matches) != 1:
        raise ValueError(f"expected one composition row for {class_name}")
    return matches[0]


def _summary_lookup(
    rows: list[SummaryRow],
    catalog: str,
    class_name: str,
    variable: str,
) -> SummaryRow:
    """Return one unambiguous summary row used in explanatory prose."""

    matches = [
        row
        for row in rows
        if row.catalog == catalog
        and row.class_name == class_name
        and row.variable == variable
    ]
    if len(matches) != 1:
        key = (catalog, class_name, variable)
        raise ValueError(f"expected one summary for {key}")
    return matches[0]


def _finite_range(values: np.ndarray) -> tuple[float, float]:
    """Return the finite range needed to describe coordinate coverage."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("values must contain at least one finite measurement")
    return float(finite.min()), float(finite.max())


def composition_interpretation(
    highlum: list[CompositionRow],
    highdens: list[CompositionRow],
) -> str:
    """Explain robust-class composition and the signed fraction contrast."""

    highlum_etg = _composition_lookup(highlum, "robust_etg")
    highlum_ltg = _composition_lookup(highlum, "robust_ltg")
    highdens_etg = _composition_lookup(highdens, "robust_etg")
    highdens_ltg = _composition_lookup(highdens, "robust_ltg")
    ltg_difference = (
        highdens_ltg.fraction_of_total - highlum_ltg.fraction_of_total
    )

    return (
        f"The highlum match contains {highlum_etg.count:,} robust ETG "
        f"classifications and {highlum_ltg.count:,} robust LTG "
        f"classifications. The highdens match contains "
        f"{highdens_etg.count:,} robust ETG classifications and "
        f"{highdens_ltg.count:,} robust LTG classifications. The robust-LTG "
        f"fraction difference (highdens - highlum) is {ltg_difference:+.3%}. "
        "This is a catalogue association that may reflect source selection; "
        "it does not establish an environmental cause."
    )


def probability_interpretation(rows: list[SummaryRow]) -> str:
    """Explain MP_LTG separation using robust-class medians and IQRs."""

    statements = []
    for catalog in ("highlum", "highdens"):
        etg = _summary_lookup(rows, catalog, "robust_etg", "MP_LTG")
        ltg = _summary_lookup(rows, catalog, "robust_ltg", "MP_LTG")
        statements.append(
            f"{catalog} robust ETG classifications have median "
            f"MP_LTG={etg.median:.3f} (IQR={etg.iqr:.3f}), while robust LTG "
            f"classifications have median {ltg.median:.3f} "
            f"(IQR={ltg.iqr:.3f})."
        )

    return " ".join(statements) + (
        " This separation is expected because the robust flags are constructed "
        "from the five model outputs; it checks internal consistency rather "
        "than independently measuring classification accuracy."
    )


def dispersion_interpretation(rows: list[SummaryRow]) -> str:
    """Compare P1-P5 disagreement without calling it total uncertainty."""

    medians = {
        (catalog, class_name): _summary_lookup(
            rows,
            catalog,
            class_name,
            "P1_P5_LTG_STD",
        ).median
        for catalog in ("highlum", "highdens")
        for class_name in ("robust_etg", "robust_ltg")
    }
    largest = max(medians, key=lambda key: medians[key])
    catalog, class_name = largest

    return (
        "The largest median five-model disagreement occurs for "
        f"{catalog} {CLASS_LABELS[class_name]} classifications "
        f"({medians[largest]:.3f}). This spread measures disagreement among "
        "the five related networks, not the full uncertainty from images, "
        "training labels, or catalogue selection."
    )


def magnitude_radius_interpretation(rows: list[SummaryRow]) -> str:
    """Describe observed magnitude and pixel-radius ranges for both files."""

    statements = []
    for catalog in ("highlum", "highdens"):
        magnitude = _summary_lookup(rows, catalog, "all_valid", "MAG_AUTO_R")
        radius = _summary_lookup(rows, catalog, "all_valid", "FLUX_RADIUS_R")
        statements.append(
            f"{catalog} spans apparent magnitudes {magnitude.minimum:.2f} to "
            f"{magnitude.maximum:.2f} and has median half-light radius "
            f"{radius.median:.2f} pixels"
        )

    return "; ".join(statements) + (
        ". These are observed image-plane quantities; distance, seeing, and "
        "selection can shape the pattern, so pixel radius is not physical size."
    )


def brightness_size_class_interpretation(rows: list[SummaryRow]) -> str:
    """Quantify robust-class median radius contrasts in each catalogue."""

    statements = []
    for catalog in ("highlum", "highdens"):
        etg = _summary_lookup(rows, catalog, "robust_etg", "FLUX_RADIUS_R")
        ltg = _summary_lookup(rows, catalog, "robust_ltg", "FLUX_RADIUS_R")
        difference = ltg.median - etg.median
        statements.append(
            f"In {catalog}, the robust-LTG median half-light radius is "
            f"{difference:+.2f} pixels relative to the robust-ETG median"
        )

    return "; ".join(statements) + (
        ". The separation and overlap arise in shared imaging measurements, "
        "so they are descriptive rather than an independent morphology test."
    )


def faintness_interpretation(
    catalog: str,
    summary: BinnedSummary,
    minimum_count: int,
) -> str:
    """Describe endpoints only among bins meeting an explicit occupancy rule."""

    populated = np.flatnonzero(eligible_binned_bins(summary, minimum_count))
    if populated.size < 2:
        return (
            f"{catalog} has fewer than two magnitude bins with at least "
            f"{minimum_count:,} rows, so this run cannot describe a stable "
            "probability trend with faintness."
        )

    first = int(populated[0])
    last = int(populated[-1])
    return (
        f"Using bins with at least {minimum_count:,} rows, {catalog} has "
        f"median MP_LTG={summary.median[first]:.3f} (n={summary.count[first]:,}) "
        f"in the brightest eligible bin and {summary.median[last]:.3f} "
        f"(n={summary.count[last]:,}) in the faintest eligible bin. This "
        "describes model-output structure only; these matched samples have no "
        "independent truth labels with which to measure an accuracy change."
    )


def orientation_interpretation(rows: list[SummaryRow]) -> str:
    """Compare edge-on medians while preserving their orientation meaning."""

    statements = []
    for catalog in ("highlum", "highdens"):
        etg = _summary_lookup(rows, catalog, "robust_etg", "MP_EdgeOn")
        ltg = _summary_lookup(rows, catalog, "robust_ltg", "MP_EdgeOn")
        statements.append(
            f"{catalog}: ETG median={etg.median:.3f}, "
            f"LTG median={ltg.median:.3f}"
        )

    return "; ".join(statements) + (
        ". MP_EdgeOn is an orientation output, not a third morphology class; "
        "projection can hide disk and spiral structure."
    )


def sky_interpretation(
    highlum_ra: np.ndarray,
    highlum_dec: np.ndarray,
    highdens_ra: np.ndarray,
    highdens_dec: np.ndarray,
) -> str:
    """Report coordinate extents and explain why visible holes are ambiguous."""

    highlum_ra_range = _finite_range(highlum_ra)
    highlum_dec_range = _finite_range(highlum_dec)
    highdens_ra_range = _finite_range(highdens_ra)
    highdens_dec_range = _finite_range(highdens_dec)

    return (
        f"highlum covers RA {highlum_ra_range[0]:.2f}-"
        f"{highlum_ra_range[1]:.2f} deg and DEC "
        f"{highlum_dec_range[0]:.2f}-{highlum_dec_range[1]:.2f} deg; highdens "
        f"covers RA {highdens_ra_range[0]:.2f}-{highdens_ra_range[1]:.2f} deg "
        f"and DEC {highdens_dec_range[0]:.2f}-"
        f"{highdens_dec_range[1]:.2f} deg. Holes can mark footprint boundaries "
        "or bright-star masks and must not be interpreted as physical galaxy "
        "underdensities."
    )


def separation_interpretation(
    highlum: np.ndarray,
    highdens: np.ndarray,
    maximum_arcsec: float,
) -> str:
    """Compare valid separation percentiles and enforce the angular limit."""

    prepared = []
    for values in (highlum, highdens):
        numeric = np.asarray(values, dtype=float)
        valid = (
            np.isfinite(numeric)
            & (numeric >= 0)
            & (numeric <= maximum_arcsec)
        )
        selected = numeric[valid]
        if not selected.size:
            raise ValueError("separation arrays must contain valid values")
        prepared.append(selected)

    highlum_valid, highdens_valid = prepared
    return (
        "The median (95th percentile) separation is "
        f"{np.median(highlum_valid):.4g} "
        f"({np.percentile(highlum_valid, 95):.4g}) arcsec for highlum and "
        f"{np.median(highdens_valid):.4g} "
        f"({np.percentile(highdens_valid, 95):.4g}) arcsec for highdens. "
        f"All validated rows remain within the adopted {maximum_arcsec:g} "
        "arcsec angular tolerance, which is not a physical distance."
    )


def radius_cut_interpretation(
    highlum_radius: np.ndarray,
    highdens_radius: np.ndarray,
    maximum_pixels: float,
) -> str:
    """Report the exact effect of the advisor's provisional radius cut."""

    removed = []
    for radius in (highlum_radius, highdens_radius):
        values = np.asarray(radius, dtype=float)
        beyond_cut = np.isfinite(values) & (values >= maximum_pixels)
        removed.append(int(np.count_nonzero(beyond_cut)))

    highlum_word = "row" if removed[0] == 1 else "rows"
    highdens_word = "row" if removed[1] == 1 else "rows"
    return (
        f"The provisional FLUX_RADIUS_R < {maximum_pixels:g} check removes "
        f"{removed[0]:,} highlum {highlum_word} and {removed[1]:,} highdens "
        f"{highdens_word}. It remains a diagnostic and is not applied to the "
        "primary statistics until its intended scientific meaning is confirmed."
    )
