# -*- coding: utf-8 -*-
"""
This submodule implements a command line interface for pareidolia's pattern
change detection pipeline.
"""
import click
import numpy as np
import chromosight.kernels as ck
from .. import hic_utils as pah
from .. import __version__


@click.command()
@click.option(
    "--bed2d-file",
    "-b",
    default=None,
    show_default=True,
    help=(
        "Optional bed2d file containing pattern positions where changes should"
        " be measured (instead of detecting)."
    ),
    type=click.Path(exists=True),
)
@click.option(
    "--kernel",
    "-k",
    default="loops",
    show_default=True,
    help=(
        "A kernel name or a tab-separated text file containing a square kernel"
        " matrix. Valid kernel names are: loops, borders, centromeres,"
        " hairpins."
    ),
)
@click.option(
    "--region",
    "-r",
    default=None,
    show_default=True,
    help=(
        "Optional comma-separated list of regions in UCSC format (e.g."
        " chr1:1000-40000) at which detection should operate."
    ),
)
@click.option(
    "--max-dist",
    "-M",
    default=None,
    show_default=True,
    help=(
        "Maximum interaction distance (in basepairs) at which patterns should"
        " be detected. Reduce to accelerate detection and reduce memory"
        " usage."
    ),
    type=int,
)
@click.option(
    "--pearson",
    "-p",
    default=None,
    show_default=True,
    help=(
        "Threshold to apply when detecting pattern changes. "
        "A default value is selected based on the kernel."
    ),
    type=float,
)
@click.option(
    "--density",
    "-D",
    default=0.10,
    show_default=True,
    help=(
        "Minimum proportion of nonzero pixels required to consider a region. "
        "Smaller values allows lower coverage regions, but "
        "increase false positives."
    ),
    type=float,
)
@click.option(
    "--cnr",
    "-c",
    default=0.3,
    show_default=True,
    help=(
        "Contrast-to-noise-ratio threshold used to filter out positions with "
        "high technical variations relative to biological variations."
    ),
    type=float,
)
@click.option(
    "--no-subsample",
    "-S",
    is_flag=True,
    help="Disable subsampling of input matrices to the same coverage.",
)
@click.option(
    "--no-filter",
    "-F",
    is_flag=True,
    help=(
        "Completely disable pearson, cnr and density filtering. Mostly "
        "for debugging. All input positions are returned, but results "
        "will be noisy."
    ),
)
@click.option(
    "--n-cpus",
    "-n",
    default=1,
    show_default=False,
    help=(
        "Number of CPUs to use for parallel tasks. It is recommended to set "
        "at most to the number of input samples."
    ),
)
@click.argument("cool_files", type=str)
@click.argument("conditions", type=str)
@click.argument("outfile", type=str)
@click.version_option(version=__version__)
def pareidolia_cmd(
    cool_files,
    conditions,
    outfile,
    kernel,
    bed2d_file,
    region,
    max_dist,
    no_subsample,
    no_filter,
    pearson,
    density,
    cnr,
    n_cpus,
):
    """Run the pattern change detection pipeline. Given a list of cool files
    and associated conditions, compute pattern intensity change from one condition
    relative to the control. The first condition occuring in the list is the control.
    For all patterns that pass the quality filters, a differential score
    (condition - control) and a contrast-to-noise ratio are returned."""

    # Attempt to load chromosight kernel from the kernel name.
    try:
        _ = getattr(ck, kernel)
    # If the kernel name is not valid, assumes it's a custom kernel file
    except AttributeError:
        try:
            kernel = np.loadtxt(kernel)
        except OSError:
            raise ValueError(
                "kernel must either be a valid kernel name or path to a"
                " text file (see --help)."
            )
    # Disable all filters if requested
    if no_filter:
        cnr = density = None
        pearson = 0.0
    # Get lists from comma-separated items
    cool_files = _parse_cli_list(cool_files)
    conditions = _parse_cli_list(conditions)

    # Start the detection pipeline
    changes = pah.change_detection_pipeline(
        cool_files=cool_files,
        conditions=conditions,
        kernel=kernel,
        bed2d_file=bed2d_file,
        region=region,
        max_dist=max_dist,
        subsample=not no_subsample,
        pearson_thresh=pearson,
        density_thresh=density,
        cnr_thresh=cnr,
        n_cpus=n_cpus,
    )

    # Save results to text file
    changes.to_csv(outfile, sep="\t", index=False)


def _parse_cli_list(items):
    """Process a string of comma separated items into a list"""
    if items == "":
        return None
    else:
        return items.split(",")
