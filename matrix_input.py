# -*- coding: utf-8 -*-
"""
Created on Thu May 15 2025
Last Modified: Tue May 21 2025, 10:10 AM

File Name: matrix_input.py
Description: Provides functionality for reading a text file and converting 
             it into a matrix using NumPy.

Notes: Depends on the 'dependencies' module, which must expose NumPy as 'np'.

@author: jaunie
"""

import dependencies as d

def file_to_matrix(filename: str) -> 'd.np.ndarray':
    """
    Reads a text file and converts its contents into a NumPy matrix.

    Parameters:
        filename : str
            Path to the input file containing numeric data.

    Returns:
        numpy.ndarray
            A 2D NumPy array representing the contents of the file.

    Raises:
        OSError
            If the file cannot be opened or read.
        ValueError
            If the file contents cannot be parsed into a float matrix.

    Example:
        >>> matrix = file_to_matrix("inputs/data.txt")
    """
    matrix = d.np.loadtxt(filename, dtype=float)
    return matrix
