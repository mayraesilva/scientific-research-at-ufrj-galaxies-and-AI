"""Reusable pass/fail gates for matched morphology catalogues."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .selection import valid_value_mask, validate_separation


@dataclass(frozen=True)
class CatalogueQuality:
    """One complete audit row for a catalogue that passed every gate."""

    catalogue: str
    rows: int
    available_columns: int
    invalid_separations: int
    invalid_coordinates: int
    invalid_probabilities: int
    invalid_flags: int
    duplicate_morphology_ids: int
    duplicate_environment_ids: int
    status: str


def assess_catalog_quality(
    catalog: str,
    expected_rows: int,
    available_columns: int,
    data: Mapping[str, np.ndarray],
    probability_columns: tuple[str, ...],
    maximum_separation_arcsec: float,
) -> CatalogueQuality:
    """Audit required domains and stop before invalid data reach analysis.

    Duplicate identifiers are failures, not informational warnings: allowing
    them into a catalogue labeled ``PASS`` would invalidate later row counts
    and over-represent repeated objects.
    """

    observed_rows = len(data["FLAG_LTG"])
    separation = validate_separation(
        data["Separation"],
        maximum_separation_arcsec,
    )
    invalid_coordinates = int(
        np.count_nonzero(~valid_value_mask(data["RA_2"], "ra_deg"))
        + np.count_nonzero(~valid_value_mask(data["DEC_2"], "dec_deg"))
    )
    invalid_probabilities = int(
        sum(
            np.count_nonzero(
                ~valid_value_mask(data[column], "probability")
            )
            for column in probability_columns
        )
    )
    invalid_flags = int(
        np.count_nonzero(~np.isin(data["FLAG_LTG"], np.arange(6)))
    )
    duplicate_morphology_ids = int(
        len(data["COADD_OBJECT_ID"])
        - len(np.unique(data["COADD_OBJECT_ID"]))
    )
    duplicate_environment_ids = int(
        len(data["object_id"]) - len(np.unique(data["object_id"]))
    )

    if observed_rows != expected_rows:
        raise ValueError(f"{catalog} row count changed while reading")
    if not separation.is_valid:
        raise ValueError(
            f"{catalog} contains matches outside "
            f"{maximum_separation_arcsec:g} arcsec"
        )
    if invalid_coordinates or invalid_probabilities or invalid_flags:
        raise ValueError(f"{catalog} failed value-domain validation")
    if duplicate_morphology_ids or duplicate_environment_ids:
        raise ValueError(f"{catalog} contains duplicate identifiers")

    return CatalogueQuality(
        catalogue=catalog,
        rows=expected_rows,
        available_columns=available_columns,
        invalid_separations=separation.invalid_count,
        invalid_coordinates=invalid_coordinates,
        invalid_probabilities=invalid_probabilities,
        invalid_flags=invalid_flags,
        duplicate_morphology_ids=duplicate_morphology_ids,
        duplicate_environment_ids=duplicate_environment_ids,
        status="PASS",
    )


def require_catalogue_baseline(
    catalog: str,
    row_count: int,
    flags: np.ndarray,
    expected_row_count: int,
    expected_flag_counts: Mapping[int, int],
) -> None:
    """Raise a runtime error when a meeting-approved baseline has drifted.

    An explicit exception works under optimized Python, unlike ``assert``.
    This strict regression gate is intended only for a baseline explicitly
    approved by the researcher, currently the smaller ``highlum`` file.
    """

    observed_flags, observed_counts = np.unique(flags, return_counts=True)
    observed_flag_counts = {
        int(flag): int(count)
        for flag, count in zip(observed_flags, observed_counts, strict=True)
    }
    if (
        row_count != expected_row_count
        or observed_flag_counts != dict(expected_flag_counts)
    ):
        raise ValueError(
            f"{catalog} baseline changed: expected {expected_row_count:,} rows "
            f"and flags {dict(expected_flag_counts)}, observed {row_count:,} "
            f"rows and flags {observed_flag_counts}"
        )
