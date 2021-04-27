pareidolia
==========

.. image:: https://img.shields.io/pypi/v/pareidolia.svg
    :target: https://pypi.python.org/pypi/pareidolia
    :alt: Latest PyPI version

.. image:: https://github.com/koszullab/pareidolia/actions/workflows/python-package.yml/badge.svg
   :target: https://github.com/koszullab/pareidolia/actions/workflows/python-package.yml
   :alt: build

.. image:: https://codecov.io/gh/koszullab/pareidolia/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/koszullab/pareidolia

Multi-sample change detection in Hi-C patterns

Pareidolia for detects changes in intensities of a specific pattern (e.g. chromatin loops and domain borders) from Hi-C maps.
It can be used to compare samples from different conditions and use multiple replicates to improve results.

This toolkit exploits `Chromosight <https://github.com/koszullab/chromosight>`_ correlation maps, allowing the same method to detect changes in different Hi-C patterns (e.g. loops or borders).

Usage
-----

Pareidolia can be used both as a python package and as a command line tool:

.. code-block:: python

  import pareidolia.hic_utils as pah
  import chromosight.kernels as ck
  pah.change_detection_pipeline(
    ["ctrl1.cool", "ctrl2.cool", "treat1.cool", "treat2.cool"],
    ["control", "control", "treatment", "treatment"],
    kernel=ck.loops,
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

Padeidolia accepts chromosight kernels as kernel names. A list of valid kernels can be displayed using `chromosight list-kernels`.
Alternatively, when using the API, an arbitrary 2D numpy array can be provided as kernels.

Installation
------------

Pareidolia is available on Pypi and can be installed using:

.. code:: bash

  pip3 install --user pareidolia


