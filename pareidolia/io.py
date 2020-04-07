
#!/usr/bin/env python3
"""
Functions used to load and save files.
cmdoret, 20200406
"""

from typing import Iterable
import cooler

def get_cools(path_list: Iterable[str]):
    """
    Load multiple cool files, ensuring they have the same resolution and
    shape.
    """
    cools = [None] * len(path_list)
    # Do not attempt loading if input is a single path
    if isinstance(cools, str):
        raise TypeError("Input must be an iterable of strings.")
    for i, cool in enumerate(path_list):
        clr = cooler.Cooler(cool)
        # Use the first cool file to define expected format
        if i == 0:
            shape = clr.shape
            binsize = clr.binsize
        # Check if format of next files is identical
        else:
            if clr.shape != shape:
                raise ValueError("Shapes are inconsistent.")
            if clr.binsize != binsize:
                raise ValueError("Resolutions are inconsistent.")
        cools[i] = clr

    return cools
