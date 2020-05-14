pareidolia
==========

.. image:: https://img.shields.io/pypi/v/pareidolia.svg
    :target: https://pypi.python.org/pypi/pareidolia
    :alt: Latest PyPI version

.. image:: https://travis-ci.com/cmdoret/pareidolia.png
   :target: https://travis-ci.com/cmdoret/pareidolia
   :alt: Latest Travis CI build status

.. image:: https://codecov.io/gh/cmdoret/pareidolia/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/cmdoret/pareidolia

Multi-sample change detection in Hi-C patterns

Pareidolia is a toolkit for detecting changes in pattern intensities from Hi-C maps. It can be used to compare samples from different conditions and use multiple replicates to improve results. 

This toolkit exploits chromosight's correlation maps, allowing the same method to detect changes in different Hi-C patterns (e.g. loops or borders).

Usage
-----

The api is currently a WIP and will be exposed through a command line interface in the future.

Installation
------------

Pareidolia is available on Pypi and can be installed using:

.. code:: bash

  pip3 install --user pareidolia


