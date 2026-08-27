"""Behavior tests for reusable matched-catalogue quality gates."""

import numpy as np
import pytest

from src.galaxy_analysis.quality import (
    assess_catalog_quality,
    require_catalogue_baseline,
)


def valid_catalogue():
    """Return a minimal valid mapping used by focused quality tests."""

    return {
        "COADD_OBJECT_ID": np.array([1, 2, 3]),
        "object_id": np.array([11, 12, 13]),
        "RA_2": np.array([10.0, 20.0, 30.0]),
        "DEC_2": np.array([-1.0, 0.0, 1.0]),
        "MP_LTG": np.array([0.1, 0.5, 0.9]),
        "FLAG_LTG": np.array([4, 4, 5]),
        "Separation": np.array([0.1, 0.2, 1.0]),
    }


def test_assess_catalog_quality_returns_a_passing_audit():
    """Catch quality rows that omit domains, duplicates, or separation checks."""

    result = assess_catalog_quality(
        catalog="sample",
        expected_rows=3,
        available_columns=7,
        data=valid_catalogue(),
        probability_columns=("MP_LTG",),
        maximum_separation_arcsec=1.0,
    )

    assert result.status == "PASS"
    assert result.rows == 3
    assert result.invalid_separations == 0
    assert result.duplicate_morphology_ids == 0
    assert result.duplicate_environment_ids == 0


def test_assess_catalog_quality_rejects_duplicate_identifiers():
    """Catch duplicate identifiers being counted but still labeled PASS."""

    data = valid_catalogue()
    data["object_id"] = np.array([11, 11, 13])

    with pytest.raises(ValueError, match="duplicate identifiers"):
        assess_catalog_quality(
            catalog="sample",
            expected_rows=3,
            available_columns=7,
            data=data,
            probability_columns=("MP_LTG",),
            maximum_separation_arcsec=1.0,
        )


def test_require_catalogue_baseline_uses_explicit_runtime_errors():
    """Catch optimized-away assertions or silent meeting-baseline drift."""

    require_catalogue_baseline(
        catalog="highlum",
        row_count=3,
        flags=np.array([4, 4, 5]),
        expected_row_count=3,
        expected_flag_counts={4: 2, 5: 1},
    )

    with pytest.raises(ValueError, match="highlum baseline changed"):
        require_catalogue_baseline(
            catalog="highlum",
            row_count=4,
            flags=np.array([4, 4, 5]),
            expected_row_count=3,
            expected_flag_counts={4: 2, 5: 1},
        )
