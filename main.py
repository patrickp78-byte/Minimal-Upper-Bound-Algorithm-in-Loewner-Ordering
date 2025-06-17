# -*- coding: utf-8 -*-
"""
Created on Thu May 15 2025
Last Modified: Tue May 29 2025, 2:33 PM

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
import dependencies as d
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
    use_rand = False
    rand_trials = 1
    args = []

    # Parse args
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--rand":
            use_rand = True
            if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
                rand_trials = int(sys.argv[i + 1])
                i += 1  # skip the number
            else:
                rand_trials = 1
        else:
            args.append(sys.argv[i])
        i += 1

    for arg in args:
        filename = 'inputs/' + arg
        print(f"\nProcessing: {filename}")
        matrix = mf.file_to_matrix(filename)
        matrices.append(matrix)
        print("Matrix: \n", matrix, "\n")
        eigs = mf.get_eigens(matrix)
        evals.append(eigs)
        labels.append(arg)

    if len(matrices) > 1:
        results = []
        for i in range(rand_trials):
            print(f"trial {i}")
            steps = 0
            try:
                is_min_upbd, M = mf.minimality_check(matrices, steps, use_rand)
                eigs = mf.get_eigens(M)
                label = f"answer_{i}" if is_min_upbd else f"failure_{i}"
            except Exception as e:
                print(f"Trial {i} failed with error: {e}")
                # Placeholder: zero matrix with same shape as original
                M = d.np.zeros_like(matrices[0])
                eigs = mf.get_eigens(M)
                label = f"error_{i}"
            
            results.append(M)
            evals.append(eigs)
            labels.append(label)

        results = d.np.array(results)
        print(f"results = \n{results}")

    gf.plot_ellipse(evals, labels)

if __name__ == "__main__":
    main()