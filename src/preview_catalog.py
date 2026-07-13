"""Display the first rows of the DES morphological FITS catalog."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table


# Build the default path relative to this script. This keeps the script working
# even when it is executed from a directory other than the project root.
DEFAULT_CATALOG = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "DES_DR1_CNN_morphological_catalog.fit"
)


def first_table_hdu(hdul: fits.HDUList) -> int:
    """Return the index of the first table extension in a FITS file."""
    # A FITS file may contain images and several table extensions. Search for
    # the first extension that Astropy identifies as an ASCII or binary table.
    for index, hdu in enumerate(hdul):
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            return index

    # Give a clear error instead of failing later while trying to access rows.
    raise ValueError("The FITS file does not contain a table extension.")


def preview_catalog(path: Path, row_count: int) -> Table:
    """Read only the requested initial rows using FITS memory mapping."""
    # Memory mapping lets the operating system access data directly from disk,
    # so the entire 3.8 GB catalog does not need to be copied into RAM.
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        # Locate the catalog table rather than assuming a fixed HDU number.
        table_hdu = hdul[first_table_hdu(hdul)]

        # Slice before creating the Astropy Table so only the requested rows
        # become part of the preview object.
        rows = Table(table_hdu.data[:row_count])

    return rows


def plot_magnitude_vs_flux_radius(rows: Table) -> plt.Figure:
    """Create a scatter plot of R-band magnitude against flux radius."""
    # Convert the two catalog columns to NumPy arrays so invalid values can be
    # removed before plotting.
    magnitude = np.asarray(rows["MAG_AUTO_R"], dtype=float)
    flux_radius = np.asarray(rows["FLUX_RADIUS_R"], dtype=float)
    valid = np.isfinite(magnitude) & np.isfinite(flux_radius)

    if not np.any(valid):
        raise ValueError("No valid MAG_AUTO_R and FLUX_RADIUS_R pairs to plot.")

    # Each point represents one galaxy from the rows selected for the preview.
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(
        magnitude[valid],
        flux_radius[valid],
        alpha=0.7,
        edgecolors="none",
    )
    axis.set_title("R-band magnitude versus flux radius")
    axis.set_xlabel("MAG_AUTO_R")
    axis.set_ylabel("FLUX_RADIUS_R")
    axis.grid(alpha=0.25)

    # Astronomical magnitudes run backwards: smaller values are brighter.
    axis.invert_xaxis()
    figure.tight_layout()
    return figure


def main() -> None:
    # Define command-line options. The catalog path is optional because this
    # project already has a known default location for the data file.
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "catalog",
        nargs="?",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"FITS catalog path (default: {DEFAULT_CATALOG})",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=50,
        help="Number of rows to display (default: 50)",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        help="Save the graph to this path instead of opening a graph window",
    )
    args = parser.parse_args()

    # Validate user input before attempting to open the large catalog.
    if args.rows < 1:
        parser.error("--rows must be at least 1")
    if not args.catalog.is_file():
        parser.error(f"catalog not found: {args.catalog}")

    # Load and print the preview. Unlimited display width and lines ensure that
    # Astropy does not replace part of the requested output with ellipses.
    rows = preview_catalog(args.catalog, args.rows)
    print(f"Catalog: {args.catalog}")
    print(f"Showing {len(rows)} rows and {len(rows.colnames)} columns\n")

    # Number columns from 1 to make it easy to refer to them during analysis.
    print("Column numbers and names:")
    for column_number, column_name in enumerate(rows.colnames, start=1):
        print(f"{column_number:>3}: {column_name}")

    print("\nFirst rows:")
    rows.pprint(max_lines=-1, max_width=-1)

    # Create the requested MAG_AUTO_R versus FLUX_RADIUS_R graph. By default it
    # opens in a window; --save-plot makes non-interactive use possible.
    print("\nGraph: MAG_AUTO_R versus FLUX_RADIUS_R")
    figure = plot_magnitude_vs_flux_radius(rows)
    if args.save_plot:
        args.save_plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save_plot, dpi=150)
        print(f"Graph saved to: {args.save_plot}")
    else:
        plt.show()


if __name__ == "__main__":
    # Run main only when this file is executed directly, not when it is imported.
    main()
