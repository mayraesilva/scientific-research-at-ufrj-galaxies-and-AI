import numpy as np
import pytest
from astropy.io import fits

from src.galaxy_analysis.catalog import (
    find_table_hdu,
    inspect_catalog,
    random_indices,
    read_columns,
)


def test_find_table_hdu_returns_first_binary_table(synthetic_crossmatch_path):
    with fits.open(synthetic_crossmatch_path) as hdul:
        assert find_table_hdu(hdul) == 1


def test_find_table_hdu_rejects_image_only_fits():
    with fits.HDUList([fits.PrimaryHDU()]) as hdul:
        with pytest.raises(ValueError, match="does not contain a table HDU"):
            find_table_hdu(hdul)


def test_inspect_catalog_reports_schema_without_loading_all_columns(synthetic_crossmatch_path):
    schema = inspect_catalog(synthetic_crossmatch_path)
    assert schema.hdu_index == 1
    assert schema.extname == "JOINED"
    assert schema.row_count == 7
    assert schema.column_count == 21
    assert schema.column_units["Separation"] == "arcsec"


def test_read_columns_returns_only_requested_columns(synthetic_crossmatch_path):
    result = read_columns(synthetic_crossmatch_path, ("FLAG_LTG", "MP_LTG"))
    assert tuple(result) == ("FLAG_LTG", "MP_LTG")
    np.testing.assert_array_equal(result["FLAG_LTG"], [0, 1, 2, 3, 4, 4, 5])


def test_read_columns_preserves_sorted_index_order(synthetic_crossmatch_path):
    result = read_columns(synthetic_crossmatch_path, ("FLAG_LTG",), np.array([1, 4, 6]))
    np.testing.assert_array_equal(result["FLAG_LTG"], [1, 4, 5])


def test_read_columns_rejects_missing_column(synthetic_crossmatch_path):
    with pytest.raises(KeyError, match="missing required FITS columns: UNKNOWN"):
        read_columns(synthetic_crossmatch_path, ("UNKNOWN",))


def test_random_indices_are_unique_sorted_and_reproducible():
    first = random_indices(1_000, 100, 20260713)
    second = random_indices(1_000, 100, 20260713)
    np.testing.assert_array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 100
    assert np.all(first[:-1] < first[1:])


def test_random_indices_caps_sample_to_population():
    result = random_indices(3, 100, 20260713)
    np.testing.assert_array_equal(result, [0, 1, 2])


@pytest.mark.parametrize("population,sample", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_random_indices_rejects_nonpositive_sizes(population, sample):
    with pytest.raises(ValueError, match="must be at least 1"):
        random_indices(population, sample, 20260713)
