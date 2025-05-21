# -*- coding: utf-8 -*-
"""
Created on Thu May 15 21:48:02 2025
Last Modified: Tue May 21 2025, 10:00 AM

File Name: main.py
Description: Entry point for processing one or more matrix input files. 
             Reads each specified file from the 'inputs' directory, converts 
             its contents into a matrix, and prints the matrix to the console.

Notes: Depends on the 'matrix_input' module which must contain a 'file_to_matrix' function.

@author: patrick, jaunie
"""

import sys
import matrix_input as mi

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

    for arg in sys.argv[1:]:
        filename = 'inputs/' + arg
        print(f"\nProcessing: {filename}")
        matrix = mi.file_to_matrix(filename)
        print("Matrix: \n", matrix, "\n")

if __name__ == "__main__":
    main()