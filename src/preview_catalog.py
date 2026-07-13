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

# Always use the same output path so rerunning the script replaces the previous
# graph instead of creating duplicate image files.
DEFAULT_GRAPH = (
    Path(__file__).resolve().parents[1]
    / "graphs"
    / "mag_auto_r_vs_flux_radius_r.png"
)
DEFAULT_HEATMAP = (
    Path(__file__).resolve().parents[1]
    / "graphs"
    / "mag_auto_r_vs_flux_radius_r_heatmap.png"
)
DEFAULT_CONFIDENCE_HISTOGRAM = (
    Path(__file__).resolve().parents[1]
    / "graphs"
    / "neural_network_confidence_histogram.png"
)
DEFAULT_MORPHOLOGY_RELATION = (
    Path(__file__).resolve().parents[1]
    / "graphs"
    / "magnitude_size_by_morphology.png"
)
DEFAULT_FAINTNESS_BIAS = (
    Path(__file__).resolve().parents[1]
    / "graphs"
    / "classification_bias_vs_faintness.png"
)
DEFAULT_EDGE_ON_CORRELATION = (
    Path(__file__).resolve().parents[1]
    / "graphs"
    / "edge_on_disk_correlation.png"
)
DEFAULT_SKY_MAP = (
    Path(__file__).resolve().parents[1]
    / "graphs"
    / "sky_distribution_map.png"
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


def read_catalog_columns(
    path: Path,
    column_names: tuple[str, ...],
    row_count: int,
) -> dict[str, np.ndarray]:
    """Read selected FITS columns without copying the entire catalog."""
    # Selecting columns before copying avoids loading all 19 catalog columns
    # when the larger analysis sample is used to create graphs.
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        table_data = hdul[first_table_hdu(hdul)].data
        columns = {
            name: np.array(table_data[name][:row_count], copy=True)
            for name in column_names
        }

    return columns


def valid_graph_values(
    magnitude: np.ndarray,
    flux_radius: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove rows containing invalid magnitude or flux-radius values."""
    valid = np.isfinite(magnitude) & np.isfinite(flux_radius)
    if not np.any(valid):
        raise ValueError("No valid MAG_AUTO_R and FLUX_RADIUS_R pairs to plot.")

    return magnitude[valid], flux_radius[valid]


def plot_magnitude_vs_flux_radius(rows: Table) -> plt.Figure:
    """Create a scatter plot of R-band magnitude against flux radius."""
    # Convert the two catalog columns to NumPy arrays so invalid values can be
    # removed before plotting.
    magnitude = np.asarray(rows["MAG_AUTO_R"], dtype=float)
    flux_radius = np.asarray(rows["FLUX_RADIUS_R"], dtype=float)
    magnitude, flux_radius = valid_graph_values(magnitude, flux_radius)

    # Each point represents one galaxy from the rows selected for the preview.
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(
        magnitude,
        flux_radius,
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


def plot_density_heatmap(
    magnitude: np.ndarray,
    flux_radius: np.ndarray,
) -> plt.Figure:
    """Create a hexbin heatmap showing galaxy density in parameter space."""
    magnitude, flux_radius = valid_graph_values(magnitude, flux_radius)

    # Hexagonal bins show dense regions clearly when individual scatter points
    # would overlap. Logarithmic coloring also preserves low-density structure.
    figure, axis = plt.subplots(figsize=(9, 6))
    density = axis.hexbin(
        magnitude,
        flux_radius,
        gridsize=70,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    axis.set_title("Galaxy density by R-band magnitude and flux radius")
    axis.set_xlabel("MAG_AUTO_R")
    axis.set_ylabel("FLUX_RADIUS_R")
    axis.invert_xaxis()

    colorbar = figure.colorbar(density, ax=axis)
    colorbar.set_label("Galaxy count per bin (log scale)")
    figure.tight_layout()
    return figure


def plot_neural_network_confidence_histogram(
    probabilities: np.ndarray,
    probability_column: str = "MP_LTG",
) -> plt.Figure:
    """Plot the distribution of a neural-network confidence column."""
    probabilities = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)
    probabilities = probabilities[valid]
    if probabilities.size == 0:
        raise ValueError(f"No valid probabilities found in {probability_column}.")

    # A logarithmic count axis shows both the large peaks near confident
    # predictions and the smaller population of ambiguous classifications.
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.hist(
        probabilities,
        bins=50,
        range=(0, 1),
        color="slateblue",
        edgecolor="white",
        linewidth=0.4,
        log=True,
    )
    axis.set_title(f"Neural network confidence distribution: {probability_column}")
    axis.set_xlabel(f"{probability_column} probability")
    axis.set_ylabel("Galaxy count (log scale)")
    axis.set_xlim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def plot_magnitude_size_by_morphology(
    magnitude: np.ndarray,
    flux_radius: np.ndarray,
    morphology_flag: np.ndarray,
) -> plt.Figure:
    """Plot the magnitude-size relation separated into LTG and ETG classes."""
    magnitude = np.asarray(magnitude, dtype=float)
    flux_radius = np.asarray(flux_radius, dtype=float)
    morphology_flag = np.asarray(morphology_flag)
    valid = (
        np.isfinite(magnitude)
        & np.isfinite(flux_radius)
        & np.isfinite(morphology_flag)
    )
    if not np.any(valid):
        raise ValueError("No valid magnitude, size, and morphology rows to plot.")

    magnitude = magnitude[valid]
    flux_radius = flux_radius[valid]
    morphology_flag = morphology_flag[valid].astype(int)

    # The catalogue defines even FLAG_LTG values (0, 2, 4) as ETGs and odd
    # values (1, 3, 5) as LTGs. The higher flag values indicate more robust
    # classifications, but parity always identifies the morphology class.
    is_ltg = morphology_flag % 2 == 1
    is_etg = ~is_ltg

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(
        magnitude[is_ltg],
        flux_radius[is_ltg],
        s=3,
        alpha=0.25,
        color="tab:blue",
        edgecolors="none",
        label="LTG / Spiral",
        rasterized=True,
    )
    axis.scatter(
        magnitude[is_etg],
        flux_radius[is_etg],
        s=3,
        alpha=0.25,
        color="tab:red",
        edgecolors="none",
        label="ETG / non-LTG",
        rasterized=True,
    )
    axis.set_title("Magnitude-size relation separated by morphology")
    axis.set_xlabel("MAG_AUTO_R")
    axis.set_ylabel("FLUX_RADIUS_R")
    axis.invert_xaxis()
    axis.legend(markerscale=4)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_classification_bias_against_faintness(
    magnitude: np.ndarray,
    ltg_probability: np.ndarray,
) -> plt.Figure:
    """Plot LTG probability density as a function of apparent magnitude."""
    magnitude, ltg_probability = valid_graph_values(magnitude, ltg_probability)
    in_probability_range = (ltg_probability >= 0) & (ltg_probability <= 1)
    magnitude = magnitude[in_probability_range]
    ltg_probability = ltg_probability[in_probability_range]
    if magnitude.size == 0:
        raise ValueError("No valid magnitude and MP_LTG pairs to plot.")

    # The catalog is large, so logarithmic hexbin density is more legible than
    # drawing thousands of overlapping scatter points. Magnitude increases to
    # the right here so the direction of increasing faintness remains clear.
    figure, axis = plt.subplots(figsize=(9, 6))
    density = axis.hexbin(
        magnitude,
        ltg_probability,
        gridsize=70,
        bins="log",
        mincnt=1,
        cmap="magma",
    )
    axis.set_title("LTG classification probability versus faintness")
    axis.set_xlabel("MAG_AUTO_R (fainter galaxies toward the right)")
    axis.set_ylabel("MP_LTG")
    axis.set_ylim(0, 1)
    colorbar = figure.colorbar(density, ax=axis)
    colorbar.set_label("Galaxy count per bin (log scale)")
    figure.tight_layout()
    return figure


def plot_edge_on_disk_correlation(
    ltg_probability: np.ndarray,
    edge_on_probability: np.ndarray,
) -> plt.Figure:
    """Plot the density relation between LTG and edge-on probabilities."""
    ltg_probability, edge_on_probability = valid_graph_values(
        ltg_probability,
        edge_on_probability,
    )
    valid_probability = (
        (ltg_probability >= 0)
        & (ltg_probability <= 1)
        & (edge_on_probability >= 0)
        & (edge_on_probability <= 1)
    )
    ltg_probability = ltg_probability[valid_probability]
    edge_on_probability = edge_on_probability[valid_probability]
    if ltg_probability.size == 0:
        raise ValueError("No valid MP_LTG and MP_EdgeOn pairs to plot.")

    # A density plot makes clusters near probability boundaries visible without
    # hiding them beneath a large number of overlapping scatter points.
    figure, axis = plt.subplots(figsize=(8, 7))
    density = axis.hexbin(
        ltg_probability,
        edge_on_probability,
        gridsize=65,
        bins="log",
        mincnt=1,
        cmap="cividis",
    )
    axis.set_title("Edge-on disk probability correlation")
    axis.set_xlabel("MP_LTG")
    axis.set_ylabel("MP_EdgeOn")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    colorbar = figure.colorbar(density, ax=axis)
    colorbar.set_label("Galaxy count per bin (log scale)")
    figure.tight_layout()
    return figure


def plot_sky_distribution(
    right_ascension: np.ndarray,
    declination: np.ndarray,
) -> plt.Figure:
    """Plot the catalog's sky coverage in right ascension and declination."""
    right_ascension, declination = valid_graph_values(
        right_ascension,
        declination,
    )

    # Tiny, partially transparent points reveal the survey footprint while
    # keeping dense areas from turning into a single opaque block.
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.scatter(
        right_ascension,
        declination,
        s=0.25,
        alpha=0.35,
        color="midnightblue",
        edgecolors="none",
        rasterized=True,
    )
    axis.set_title("Sky distribution of catalog galaxies")
    axis.set_xlabel("Right Ascension (degrees)")
    axis.set_ylabel("Declination (degrees)")
    axis.set_xlim(0, 360)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def save_graph(figure: plt.Figure, output_path: Path) -> Path:
    """Save a graph to a stable path, replacing an older version if present."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def create_catalog_analysis_graphs(
    catalog_path: Path,
    row_count: int = 100_000,
    heatmap_path: Path = DEFAULT_HEATMAP,
) -> dict[str, Path]:
    """Create and save the catalog-wide analysis graphs for a row sample."""
    required_columns = (
        "RA",
        "DEC",
        "MAG_AUTO_R",
        "FLUX_RADIUS_R",
        "MP_LTG",
        "MP_EdgeOn",
        "FLAG_LTG",
    )
    columns = read_catalog_columns(catalog_path, required_columns, row_count)

    # Build each figure through a dedicated reusable function. Fixed output
    # names cause subsequent runs to update rather than duplicate the graphs.
    figures_and_paths = {
        "Magnitude-radius density heatmap": (
            plot_density_heatmap(
                columns["MAG_AUTO_R"],
                columns["FLUX_RADIUS_R"],
            ),
            heatmap_path,
        ),
        "Neural network confidence histogram": (
            plot_neural_network_confidence_histogram(columns["MP_LTG"]),
            DEFAULT_CONFIDENCE_HISTOGRAM,
        ),
        "Magnitude-size relation by morphology": (
            plot_magnitude_size_by_morphology(
                columns["MAG_AUTO_R"],
                columns["FLUX_RADIUS_R"],
                columns["FLAG_LTG"],
            ),
            DEFAULT_MORPHOLOGY_RELATION,
        ),
        "Classification bias against faintness": (
            plot_classification_bias_against_faintness(
                columns["MAG_AUTO_R"],
                columns["MP_LTG"],
            ),
            DEFAULT_FAINTNESS_BIAS,
        ),
        "Edge-on disk correlation": (
            plot_edge_on_disk_correlation(
                columns["MP_LTG"],
                columns["MP_EdgeOn"],
            ),
            DEFAULT_EDGE_ON_CORRELATION,
        ),
        "Sky distribution map": (
            plot_sky_distribution(columns["RA"], columns["DEC"]),
            DEFAULT_SKY_MAP,
        ),
    }

    saved_graphs = {}
    for graph_name, (figure, output_path) in figures_and_paths.items():
        saved_graphs[graph_name] = save_graph(figure, output_path)

    return saved_graphs


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
        default=DEFAULT_GRAPH,
        help=f"Graph output path (default: {DEFAULT_GRAPH})",
    )
    parser.add_argument(
        "--analysis-rows",
        "--heatmap-rows",
        dest="analysis_rows",
        type=int,
        default=100_000,
        help="Number of rows used for catalog-wide graphs (default: 100000)",
    )
    parser.add_argument(
        "--save-heatmap",
        type=Path,
        default=DEFAULT_HEATMAP,
        help=f"Heatmap output path (default: {DEFAULT_HEATMAP})",
    )
    args = parser.parse_args()

    # Validate user input before attempting to open the large catalog.
    if args.rows < 1:
        parser.error("--rows must be at least 1")
    if args.analysis_rows < 1:
        parser.error("--analysis-rows must be at least 1")
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

    # Save the requested graph to a stable filename. Matplotlib overwrites the
    # existing PNG at this path, preventing duplicates after repeated runs.
    print("\nGraph: MAG_AUTO_R versus FLUX_RADIUS_R")
    figure = plot_magnitude_vs_flux_radius(rows)
    save_graph(figure, args.save_plot)
    print(f"Graph saved to: {args.save_plot}")

    # Use a larger sample for the catalog-wide graphs while loading only the
    # seven required FITS columns. Every graph has a stable output filename.
    print(f"\nAnalysis graphs: using up to {args.analysis_rows} catalog rows")
    saved_graphs = create_catalog_analysis_graphs(
        args.catalog,
        row_count=args.analysis_rows,
        heatmap_path=args.save_heatmap,
    )
    for graph_name, output_path in saved_graphs.items():
        print(f"{graph_name} saved to: {output_path}")


if __name__ == "__main__":
    # Run main only when this file is executed directly, not when it is imported.
    main()
