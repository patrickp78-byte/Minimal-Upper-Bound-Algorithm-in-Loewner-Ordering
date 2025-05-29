# -*- coding: utf-8 -*-
"""
Created on Thu May 15 2025
Last Modified: Tue May 23 2025, 2:38 AM

File Name: main.py
Description: Entry point for processing one or more matrix input files. 
             For each file, the script reads the matrix, prints it, computes 
             its eigenvalues and eigenvectors, and visualizes it as an ellipse 
             or ellipsoid. Also checks whether the first matrix is an upper 
             bound or a minimal upper bound.

Notes: 
    - Expects input files in the 'inputs/' directory.

@author: patrick, jaunie
"""

import sys
import matrix_functions as mf
import graphing_functions as gf

def main():
    """
    Processes one or more matrix input files provided via command-line arguments.

    Parameters:
        None (reads from sys.argv)

    Returns:
        None

    Raises:
        FileNotFoundError:
            If one of the input files does not exist in the 'inputs/' directory.
        ValueError:
            If the file contents cannot be converted into a matrix.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <filename1> [<filename2> ...]")
        return
    
    evals = [] # list of eigenvalues and eigenvectors
    labels = [] # corresponding file names
    matrices = []

    for arg in sys.argv[1:]:
        filename = 'inputs/' + arg
        print(f"\nProcessing: {filename}")
        matrix = mf.file_to_matrix(filename)
        matrices.append(matrix)
        print("Matrix: \n", matrix, "\n")
        eigs = mf.get_eigens(matrix)
        evals.append(eigs)
        labels.append(arg)

    if len(matrices) > 1:
        if mf.is_upperbound(matrices[0], matrices[1:])[0]:
            print("Matrix: \n", matrices[0], "\n is an upperbound")
        else:
            print("Matrix: \n", matrices[0], "\n is not an upperbound")
        
        if mf.minimality_check(matrices)[0]:
            print("Matrix: \n", matrices[0], "\n is a minimal upperbound")
        else:
            print("Matrix: \n", matrices[0], "\n is not a minimal upperbound")

    # gf.plot_ellipse(evals, labels)

if __name__ == "__main__":
    main()