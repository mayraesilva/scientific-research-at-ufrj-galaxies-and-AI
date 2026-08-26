"""Memory-conscious FITS catalogue inspection and column selection.

The real catalogues contain up to tens of millions of rows.  This module keeps
schema inspection separate from data loading so the notebook never needs to
materialize a complete 191-column FITS table merely to inspect it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class CatalogSchema:
    """Immutable description of one FITS table and its column metadata."""

    path: Path
    hdu_index: int
    extname: str | None
    row_count: int
    column_count: int
    column_names: tuple[str, ...]
    column_formats: dict[str, str]
    column_units: dict[str, str | None]


def find_table_hdu(hdul: fits.HDUList) -> int:
    """Return the index of the first FITS table extension.

    FITS files often have an empty primary HDU followed by a table, but that is
    a convention rather than a guarantee.  Detecting the table by HDU type
    avoids silently reading the wrong extension when file layouts differ.
    """
    for index, hdu in enumerate(hdul):
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            return index
    raise ValueError("FITS file does not contain a table HDU")


def inspect_catalog(path: Path) -> CatalogSchema:
    """Read schema metadata without copying catalogue rows.

    Memory mapping and lazy HDU loading let us audit row counts, column names,
    formats, and units before choosing which arrays are actually needed.
    """
    path = Path(path)
    # Read-only mode protects the scientific input; lazy loading avoids touching
    # large table blocks while we inspect header/column metadata.
    with fits.open(path, mode="readonly", memmap=True, lazy_load_hdus=True) as hdul:
        index = find_table_hdu(hdul)
        hdu = hdul[index]
        names = tuple(hdu.columns.names)
        return CatalogSchema(
            path=path,
            hdu_index=index,
            extname=hdu.header.get("EXTNAME"),
            row_count=int(hdu.header["NAXIS2"]),
            column_count=len(names),
            column_names=names,
            column_formats={column.name: column.format for column in hdu.columns},
            column_units={column.name: column.unit for column in hdu.columns},
        )


def read_columns(
    path: Path,
    names: tuple[str, ...],
    indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Copy only named columns, optionally restricted to selected row indices.

    Explicit column names are the primary memory-safety boundary: the
    905,291-row cross-match has 191 columns, while this analysis needs only 21.
    Arrays are copied before the FITS handle closes so returned values do not
    depend on a live memory map.
    """
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        data = hdul[find_table_hdu(hdul)].data
        # Fail before reading anything if the input schema cannot satisfy the
        # analysis contract; partial results would be misleading.
        missing = [name for name in names if name not in data.names]
        if missing:
            raise KeyError(f"missing required FITS columns: {', '.join(missing)}")
        if indices is None:
            return {name: np.array(data[name], copy=True) for name in names}
        # int64 supports indices into the 26.97-million-row parent catalogue and
        # gives Astropy one consistent advanced-indexing type.
        indices = np.asarray(indices, dtype=np.int64)
        return {name: np.array(data[name][indices], copy=True) for name in names}


def random_indices(population_size: int, sample_size: int, seed: int) -> np.ndarray:
    """Return sorted deterministic indices sampled without replacement.

    Sampling positions instead of the first N rows avoids bias from catalogue
    ordering.  Sorting happens after random selection, preserving randomness
    while making later FITS reads more sequential and reproducible.
    """
    if population_size < 1:
        raise ValueError("population_size must be at least 1")
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    size = min(population_size, sample_size)
    # ``replace=False`` is what guarantees that one source row cannot appear
    # twice in the saved extraction.
    indices = np.random.default_rng(seed).choice(population_size, size, replace=False)
    return np.sort(indices)
