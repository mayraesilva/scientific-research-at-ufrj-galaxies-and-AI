"""Turn calculated morphology summaries into cautious research explanations.

The functions in this module do not inspect figures or invent conclusions.
They receive the same numerical summaries used by the plots and describe only
those measurements.  This keeps notebook prose natural while ensuring that a
catalogue update cannot leave hard-coded numbers behind.
"""

from collections.abc import Mapping

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


def _brightness_radius_locus(
    magnitude: np.ndarray,
    radius: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Summarize the central paired locus and its bright-to-faint direction."""

    magnitude = np.asarray(magnitude, dtype=float)
    radius = np.asarray(radius, dtype=float)
    valid = np.isfinite(magnitude) & np.isfinite(radius) & (radius > 0)
    magnitude = magnitude[valid]
    radius = radius[valid]
    if not magnitude.size:
        raise ValueError("magnitude and radius need at least one valid pair")

    magnitude_low, magnitude_high = np.percentile(magnitude, [5, 95])
    radius_low, radius_high = np.percentile(radius, [5, 95])
    bright_boundary, faint_boundary = np.percentile(magnitude, [25, 75])
    bright_radius = float(np.median(radius[magnitude <= bright_boundary]))
    faint_radius = float(np.median(radius[magnitude >= faint_boundary]))
    return (
        float(magnitude_low),
        float(magnitude_high),
        float(radius_low),
        float(radius_high),
        bright_radius,
        faint_radius,
    )


def magnitude_radius_interpretation(
    highlum_magnitude: np.ndarray,
    highlum_radius: np.ndarray,
    highdens_magnitude: np.ndarray,
    highdens_radius: np.ndarray,
) -> str:
    """Describe each paired brightness-radius locus in image-plane units."""

    statements = []
    for catalog, magnitude, radius in (
        ("highlum", highlum_magnitude, highlum_radius),
        ("highdens", highdens_magnitude, highdens_radius),
    ):
        locus = _brightness_radius_locus(magnitude, radius)
        statements.append(
            f"The {catalog} central 90% locus spans MAG_AUTO_R "
            f"{locus[0]:.2f}-{locus[1]:.2f} mag and half-light radius "
            f"{locus[2]:.2f}-{locus[3]:.2f} pixels; median radius is "
            f"{locus[4]:.2f} pixels in the brightest magnitude quartile and "
            f"{locus[5]:.2f} pixels in the faintest quartile"
        )

    return "; ".join(statements) + (
        ". These are observed image-plane quantities; distance, seeing, and "
        "selection can shape the pattern, so pixel radius is not physical size."
    )


def brightness_size_class_interpretation(rows: list[SummaryRow]) -> str:
    """Quantify both coordinates of each robust-class median locus."""

    statements = []
    for catalog in ("highlum", "highdens"):
        etg_magnitude = _summary_lookup(
            rows, catalog, "robust_etg", "MAG_AUTO_R"
        )
        ltg_magnitude = _summary_lookup(
            rows, catalog, "robust_ltg", "MAG_AUTO_R"
        )
        etg_radius = _summary_lookup(
            rows, catalog, "robust_etg", "FLUX_RADIUS_R"
        )
        ltg_radius = _summary_lookup(
            rows, catalog, "robust_ltg", "FLUX_RADIUS_R"
        )
        magnitude_difference = ltg_magnitude.median - etg_magnitude.median
        radius_difference = ltg_radius.median - etg_radius.median
        statements.append(
            f"In {catalog}, the robust-LTG median locus is "
            f"{magnitude_difference:+.2f} magnitude (positive is fainter) "
            f"and {radius_difference:+.2f} pixels relative to robust ETG"
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


def _orientation_structure(data: Mapping[str, np.ndarray]) -> tuple[float, float, float]:
    """Measure edge-on floor concentration and low/high-LTG medians."""

    ltg = np.asarray(data["MP_LTG"], dtype=float)
    edgeon = np.asarray(data["MP_EdgeOn"], dtype=float)
    valid = np.isfinite(ltg) & np.isfinite(edgeon)
    valid &= (ltg >= 0) & (ltg <= 1) & (edgeon >= 0) & (edgeon <= 1)
    ltg = ltg[valid]
    edgeon = edgeon[valid]
    low = edgeon[ltg <= 0.2]
    high = edgeon[ltg >= 0.8]
    if not edgeon.size or not low.size or not high.size:
        raise ValueError("orientation summary needs low- and high-LTG pairs")
    return (
        float(np.mean(edgeon <= 0.05)),
        float(np.median(low)),
        float(np.median(high)),
    )


def orientation_interpretation(
    rows: list[SummaryRow],
    highlum: Mapping[str, np.ndarray],
    highdens: Mapping[str, np.ndarray],
) -> str:
    """Describe the joint LTG-edge-on density and robust-class medians."""

    statements = []
    for catalog, data in (("highlum", highlum), ("highdens", highdens)):
        etg = _summary_lookup(rows, catalog, "robust_etg", "MP_EdgeOn")
        ltg = _summary_lookup(rows, catalog, "robust_ltg", "MP_EdgeOn")
        floor_fraction, low_median, high_median = _orientation_structure(data)
        statements.append(
            f"{catalog} places {floor_fraction:.1%} of valid rows at "
            f"MP_EdgeOn <= 0.05; median MP_EdgeOn is {low_median:.3f} for "
            f"MP_LTG <= 0.2 and {high_median:.3f} for MP_LTG >= 0.8. "
            f"Within robust flags, ETG median={etg.median:.3f} and "
            f"LTG median={ltg.median:.3f}"
        )

    return "; ".join(statements) + (
        ". MP_EdgeOn is an orientation output, not a third morphology class; "
        "projection can hide disk and spiral structure."
    )


def _ra_arc_description(catalog: str, values: np.ndarray) -> str:
    """Describe the shortest circular RA arc without implying all-sky coverage."""

    finite = np.asarray(values, dtype=float)
    finite = np.sort(np.mod(finite[np.isfinite(finite)], 360.0))
    if not finite.size:
        raise ValueError("RA needs at least one finite coordinate")
    if finite.size == 1:
        return f"{catalog} occupies RA {finite[0]:.2f} deg"

    circular_gaps = np.diff(np.concatenate([finite, [finite[0] + 360.0]]))
    gap_index = int(np.argmax(circular_gaps))
    start = float(finite[(gap_index + 1) % finite.size])
    end = float(finite[gap_index])
    if start <= end:
        return f"{catalog} occupies RA {start:.2f}-{end:.2f} deg"
    return (
        f"{catalog} occupies RA {start:.2f} deg through RA=0 deg "
        f"to {end:.2f} deg"
    )


def sky_interpretation(
    highlum_ra: np.ndarray,
    highlum_dec: np.ndarray,
    highdens_ra: np.ndarray,
    highdens_dec: np.ndarray,
) -> str:
    """Report coordinate extents and explain why visible holes are ambiguous."""

    highlum_dec_range = _finite_range(highlum_dec)
    highdens_dec_range = _finite_range(highdens_dec)

    return (
        f"{_ra_arc_description('highlum', highlum_ra)} and DEC "
        f"{highlum_dec_range[0]:.2f}-{highlum_dec_range[1]:.2f} deg; highdens "
        f"{_ra_arc_description('highdens', highdens_ra).removeprefix('highdens ')} "
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
        "The median, 95th percentile, and maximum separation are "
        f"{np.median(highlum_valid):.4g}, "
        f"{np.percentile(highlum_valid, 95):.4g}, and "
        f"{np.max(highlum_valid):.4g} arcsec for highlum; "
        f"{np.median(highdens_valid):.4g}, "
        f"{np.percentile(highdens_valid, 95):.4g}, and "
        f"{np.max(highdens_valid):.4g} arcsec for highdens. "
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
