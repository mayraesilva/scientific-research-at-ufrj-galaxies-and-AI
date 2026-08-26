"""Memory-conscious FITS catalogue inspection and selection."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class CatalogSchema:
    path: Path
    hdu_index: int
    extname: str | None
    row_count: int
    column_count: int
    column_names: tuple[str, ...]
    column_formats: dict[str, str]
    column_units: dict[str, str | None]


def find_table_hdu(hdul: fits.HDUList) -> int:
    """Return the index of the first FITS table extension."""
    for index, hdu in enumerate(hdul):
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            return index
    raise ValueError("FITS file does not contain a table HDU")


def inspect_catalog(path: Path) -> CatalogSchema:
    """Read schema metadata without copying catalogue rows."""
    path = Path(path)
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
    """Copy only requested columns, optionally at selected row indices."""
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        data = hdul[find_table_hdu(hdul)].data
        missing = [name for name in names if name not in data.names]
        if missing:
            raise KeyError(f"missing required FITS columns: {', '.join(missing)}")
        if indices is None:
            return {name: np.array(data[name], copy=True) for name in names}
        indices = np.asarray(indices, dtype=np.int64)
        return {name: np.array(data[name][indices], copy=True) for name in names}


def random_indices(population_size: int, sample_size: int, seed: int) -> np.ndarray:
    """Return sorted, unique, deterministic indices sampled without replacement."""
    if population_size < 1:
        raise ValueError("population_size must be at least 1")
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    size = min(population_size, sample_size)
    indices = np.random.default_rng(seed).choice(population_size, size, replace=False)
    return np.sort(indices)
