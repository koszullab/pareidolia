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
)
@click.option(
    "--kernel",
    "-k",
    default='loops',
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
        " chr1:1000-40000) at which detection should operage."
    ),
)
@click.option(
    "--max-dist",
    "-M",
    default=1000000,
    show_default=True,
    help=(
        "Maximum interaction distance (in basepairs) at which patterns should"
        " be detected. Reduce to accelerate detection and reduce memory"
        " usage."
    ),
)
@click.option(
    "--perc-thresh",
    "-p",
    default=95,
    show_default=True,
    help=(
        "Threshold to apply when detecting pattern changes."
    ),
)
@click.option(
    "--no-subsample",
    "-s",
    default=False,
    show_default=True,
    help="Disable subsampling of input matrices to the same coverage.",
)
@click.option(
    "--n-cpus",
    "-n",
    default=1,
    show_default=False,
    help="Number of CPUs to use for parallel tasks",
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
    perc_thresh,
    n_cpus,
):
    """Run the pattern change detection pipeline"""

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
    # Get lists from comma-separated items
    cool_files = _parse_cli_list(cool_files)
    conditions = _parse_cli_list(conditions)
    changes = pah.change_detection_pipeline(
        cool_files=cool_files,
        conditions=conditions,
        kernel=kernel,
        bed2d_file=bed2d_file,
        region=region,
        max_dist=max_dist,
        subsample=not no_subsample,
        percentile_thresh=perc_thresh,
        n_cpus=n_cpus,
    )
    changes.to_csv(outfile, sep='\t', index=False)


def _parse_cli_list(items):
    """Process a string of comma separated items into a list"""
    if items == "":
        return None
    else:
        return items.split(",")
