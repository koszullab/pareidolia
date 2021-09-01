pareidolia
==========

.. image:: https://img.shields.io/pypi/v/pareidolia.svg
    :target: https://pypi.python.org/pypi/pareidolia
    :alt: Latest PyPI version

.. image:: https://github.com/koszullab/pareidolia/actions/workflows/python-package.yml/badge.svg
   :target: https://github.com/koszullab/pareidolia/actions/workflows/python-package.yml
   :alt: build

.. image:: https://readthedocs.org/projects/pareidolia/badge/?version=latest
   :target: https://pareidolia.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://codecov.io/gh/koszullab/pareidolia/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/koszullab/pareidolia

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5062484.svg
   :target: https://doi.org/10.5281/zenodo.5062484

Multi-sample change detection in Hi-C patterns

Pareidolia detects changes in intensities for a specific pattern (e.g. chromatin loops and domain borders) from Hi-C maps.
It can be used to compare samples from different conditions and use multiple replicates to improve results.

This toolkit exploits `Chromosight <https://github.com/koszullab/chromosight>`_ correlation maps, allowing the same method to detect changes in different Hi-C patterns (e.g. loops or borders).

Installation
------------

Pareidolia is available on Pypi and can be installed using:

.. code:: bash

  pip3 install --user pareidolia

Usage
-----

Pareidolia can be used both as a python package and as a command line tool:

.. code-block:: python

  import pareidolia.hic_utils as pah
  pah.change_detection_pipeline(
    ["ctrl1.cool", "ctrl2.cool", "treat1.cool", "treat2.cool"],
    ["control", "control", "treatment", "treatment"],
    kernel='loops',
    subsample=True,
    n_cpus=8,
  )

We can also use the CLI to execute the same instruction:

.. code-block:: bash

  pareidolia -n 8 \
             -k loops \
             ctrl1.cool,ctrl2.cool,treat1.cool,treat2.cool \
             control,control,treatment,treatment \
             output.tsv

Pareidolia can either detect changes *de-novo*, or compute the change intensity at a set of input positions.
The input positions can be provided as a bed2d (=bedpe) file, containing a list of 2D genomic coordinates.
This file can be provided with the `--bed2d-file` option on the CLI, or using the `bed2d_file` parameter in the python API.

Pareidolia accepts chromosight kernels as kernel names. A list of valid kernels can be displayed using `chromosight list-kernels`.

Alternatively, when using the API, an arbitrary 2D numpy array can be provided as kernels.

The options shown below allow to customize pareidolia's behavior. These options are further discussed in the tutorial, available on the `documentation website <https://pareidolia.readthedocs.io/en/latest/TUTORIAL.html>`_ .

.. code-block::

    Usage: pareidolia [OPTIONS] COOL_FILES CONDITIONS OUTFILE
    
      Run the pattern change detection pipeline. Given a list of cool files and
      associated conditions, compute pattern intensity change from one condition
      relative to the control. The first condition occuring in the list is the
      control. For all patterns that pass the quality filters, a differential
      score (condition - control) and a contrast-to-noise ratio are returned.
    
    Options:
      -b, --bed2d-file PATH   Optional bed2d file containing pattern positions
                              where changes should be measured (instead of
                              detecting).
      -c, --cnr FLOAT         Contrast to-noise-ratio threshold used to filter out
                              positions with high technical variations relative to
                              biological variations.  [default: 1.0]
      -k, --kernel TEXT       A kernel name or a tab-separated text file
                              containing a square kernel matrix. Valid kernel
                              names are: loops, borders, centromeres, hairpins.
                              [default: loops]
      -r, --region TEXT       Optional comma-separated list of regions in UCSC
                              format (e.g. chr1:1000-40000) at which detection
                              should operate.
      -M, --max-dist INTEGER  Maximum interaction distance (in basepairs) at which
                              patterns should be detected. Reduce to accelerate
                              detection and reduce memory usage.
      -p, --pearson FLOAT     Threshold to apply when detecting pattern changes. A
                              default value is selected based on the kernel.
      -D, --density FLOAT     Minimum proportion of nonzero pixels required to
                              consider a region. Smaller values allows lower
                              coverage regions, but increase false positives.
                              [default: 0.1]
      -S, --no-subsample      Disable subsampling of input matrices to the same
                              coverage.
      -F, --no-filter         Completely disable pearson, cnr and density
                              filtering. Mostly for debugging. All input positions
                              are returned, but results will be noisy.
      -n, --n-cpus INTEGER    Number of CPUs to use for parallel tasks. It is
                              recommended to set at most to the number of input
                              samples.
      --version               Show the version and exit.
      --help                  Show this message and exit.


Algorithm
---------

Pareidolia starts by running Chromosight's convolution algorithm on each input sample to compute a matrix of correlation coefficients to the target pattern. Each position represents the similarity of the region to that pattern. For each condition, a median background is generated by averaging correlation matrices from replicates.

A differential background matrix is computed by subtracting backgrounds from the different conditions. Pareidolia then applies a series of filtering steps to discard noisy regions. Three filters are applied, each with their respective threshold:

* Pearson threshold: Only regions where at least one input sample has a pearson coefficient above this threshold are considered.
* CNR threshold: Contrast-to-noise-ratio filter to discard regions where the intra-condition variability is low compared to the inter-condition difference.
* Density threshold: Coverage-based filter to remove very sparse regions. If the proportion of non-empty pixels used to compute the correlation score is below that threshold, the value is discarded.

Each filter can be selectively disabled, or its threshold adapted using command line options.

If a list of positions was provided, the filtered differential scores are returned at those positions. Otherwise, de-novo differential pattern detection is performed using connected component labelling on the matrix as in Chromosight.

.. image:: docs/img/pareidolia_process.png


Citation
--------

If you use Pareidolia in your research, you can cite the software as follows (see the `Zenodo <https://zenodo.org/record/5062484>`_ page to cite a specific version):

Cyril Matthey-Doret. (2021, July 2). koszullab/pareidolia. Zenodo. https://doi.org/10.5281/zenodo.5062484
