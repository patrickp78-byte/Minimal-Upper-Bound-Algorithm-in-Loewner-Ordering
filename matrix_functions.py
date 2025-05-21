# -*- coding: utf-8 -*-
"""
Created on Thu May 15 2025
Last Modified: Tue May 21 2025, 10:20 AM

File Name: matrix_functions.py
Description: Contains utility functions for working with matrices, including
             reading from text files and computing eigenvalues and eigenvectors
             using NumPy.

Notes: 
    - Depends on the 'dependencies' module, which must expose NumPy as 'np'.
    - Assumes input matrices for eigenvalue decomposition are symmetric.

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

def get_eigens(matrix: 'd.np.ndarray') -> tuple['d.np.ndarray', 'd.np.ndarray']:
    """
    Computes the eigenvalues and eigenvectors of a symmetric matrix using NumPy.

    Parameters:
        matrix : numpy.ndarray
            A symmetric 2D array (matrix) of floats.

    Returns:
        tuple of numpy.ndarray
            - eigvals: A 1D array of eigenvalues in ascending order.
            - eigvecs: A 2D array where each column is a normalized eigenvector 
              corresponding to an eigenvalue.

    Raises:
        ValueError
            If the input is not a 2D array or not symmetric.

    Example:
        >>> eigvals, eigvecs = get_eigens(some_matrix)
    """
    eigvals, eigvecs = d.np.linalg.eigh(matrix)
    return eigvals, eigvecs
