# -*- coding: utf-8 -*-
"""
Created on Tue May 21 2025
Last Modified: Fri May 30 2025, 8:20 PM

File Name: matrix_functions.py
Description: Contains utility functions for working with matrices, including
             reading from text files and computing eigenvalues and eigenvectors
             using NumPy.

Notes: 
    - Depends on the 'dependencies' module, which must expose NumPy as 'np'.
    - Assumes input matrices for eigenvalue decomposition are symmetric.

@author: patrick, jaunie
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

def cholesky_decomposition(matrix: 'd.np.ndarray') -> bool:
    """
    Determines whether a matrix is positive definite using Cholesky decomposition.

    Parameters:
        matrix : numpy.ndarray
            A square matrix to be tested.

    Returns:
        bool
            True if the matrix is symmetric and positive definite, False otherwise.

    Notes:
        - The function first checks for symmetry using is_symmetric().
        - NumPy's cholesky() does not automatically verify symmetry.
    """
    if not is_symmetric(matrix):
        return False

    is_pos_def = False
    try:
        d.np.linalg.cholesky(matrix)
        is_pos_def = True
    except d.np.linalg.LinAlgError:
        pass
    return is_pos_def

def is_symmetric(matrix: 'd.np.ndarray') -> bool:
    """
    Checks whether a matrix is symmetric.

    Parameters:
        matrix : numpy.ndarray
            A square matrix to test.

    Returns:
        bool
            True if the matrix is symmetric, False otherwise.
    """
    return d.np.allclose(matrix, matrix.T)

def minimality_check(matrices: list['d.np.ndarray']) -> tuple[bool, 'd.np.ndarray']:
    """
    Checks whether the first matrix in the list is a minimal upper bound of the others.
    If it is an upper bound but not minimal, projects onto the orthogonal space and tries again.

    Parameters:
        matrices : list of numpy.ndarray
            A list of square matrices. The first matrix is tested against the rest.

    Returns:
        tuple:
            bool
                True if the first matrix is an upper bound and minimal, False otherwise.
            numpy.ndarray
                The combined nullspace basis matrix E from the minimal upper bound.

    Raises:
        ValueError
            If matrices are not all square or have mismatched dimensions.
    """
    M = matrices[0]
    dim = M.shape[0]

    step_1, upperbd_results = is_upperbound(M, matrices[1:])
    step_2, E = is_minimal(upperbd_results, dim) if step_1 else (False, None)

    if step_1:
        print(f"Matrix: \n {M} \n is an upperbound")

        if step_2:
            print(f"Matrix: \n {M} \n is a minimal upperbound")
        else:
            print(f"Matrix: \n {M} \n is not a minimal upperbound")
    else:
        print(f"Matrix: \n {M} \n is not an upperbound")

    if step_1 and not step_2:
        # Generate new M from projection and try again
        new_M = minimize_upperbound(M, upperbd_results, E, dim)
        print(f"New Matrix = \n {new_M} \n")
        return minimality_check([new_M] + matrices[1:])

    return step_1 and step_2, M

def is_upperbound(M: 'd.np.ndarray', matrices: list['d.np.ndarray']) -> tuple[bool, list['d.np.ndarray']]:
    """
    Determines if matrix M is an upper bound for all matrices in the list.

    Parameters:
        M : numpy.ndarray
            The candidate upper bound matrix.
        matrices : list of numpy.ndarray
            Matrices to compare against.

    Returns:
        tuple
            - bool: True if M is an upper bound.
            - list of numpy.ndarray: List of (M - Ai) matrices that passed the test.
    """
    upperbd_results = []

    for matrix in matrices:
        this_matrix = M - matrix
        if cholesky_decomposition(this_matrix) or d.np.all(d.np.linalg.eigvalsh(this_matrix) >= -1e-7):
            upperbd_results.append(this_matrix)
        else:
            return False, []

    return True, upperbd_results

def is_minimal(upperbd_results: list['d.np.ndarray'], dim: int) -> tuple[bool, list['d.np.ndarray']]:
    """
    Checks whether the nullspaces of the differences (M - Ai) span the full space F^n.

    Parameters:
    upperbd_results : list of numpy.ndarray
        List of matrices resulting from M - Ai.
    dim : int
        Dimension of the space (should equal matrix size).

    Returns:
    tuple
        bool
            True if the nullspaces span the full space, False otherwise.
        numpy.ndarray
            The combined nullspace basis matrix E.

    Raises:
    ValueError
        If any matrix in the list is not square or dimensions are inconsistent.

    Example:
    Returns (True, E) if the nullspaces of differences span R^n.
    """
    null_bases = []

    for upbd_matrix in upperbd_results:
        null_space = d.sc.linalg.null_space(upbd_matrix, rcond=1e-7)
        if null_space.size > 0:
            null_bases.append(null_space)

    if not null_bases:
        return False, d.np.empty((dim, 0))
    
    E = d.np.hstack(null_bases)
    rank = d.np.linalg.matrix_rank(E)

    return rank >= dim, E

def minimize_upperbound(M: 'd.np.ndarray', upperbd_results: list['d.np.ndarray'], E: 'd.np.ndarray', dim: int) -> 'd.np.ndarray':
    """
    Finds a vector in the orthogonal complement of E, and uses it to project M downward
    to attempt finding a minimal upper bound.

    Parameters:
        M : numpy.ndarray
            The current upper bound matrix.
        upperbd_results : list of numpy.ndarray
            The list of (M - Ai) matrices used in upper bound check.
        E : numpy.ndarray
            Matrix whose columns are the nullspace basis vectors.
        dim : int
            Dimension of the space (should equal matrix size).

    Returns:
        numpy.ndarray
            A new matrix M - projection that is closer to a minimal upper bound.

    Raises:
        RuntimeError
            If E_perp is empty and no projection direction is found.
    """
    if E is None or E.shape[1] >= dim:
        raise RuntimeError("Cannot compute projection direction; E spans full space or is None.")

    # Compute orthogonal complement of E
    E_perp = d.sc.linalg.null_space(E.T, rcond=1e-7)

    if E_perp.shape[1] == 0:
        raise RuntimeError("No orthogonal direction found; E_perp is empty.")

    # Pick a unit vector in E_perp
    e = E_perp[:, 0]
    e = e / d.np.linalg.norm(e)
    e = e.reshape(-1, 1)

    # Estimate minimal eigenvalue of the difference matrices, (find an eigen big enough)
    eigs = d.np.linalg.eigvalsh(upperbd_results).reshape(-1)
    safe_eigs = [ei for ei in eigs if ei >= 1e-7]
    lambda_min = min(safe_eigs)

    # Project and update M
    e_star = e.reshape(1, -1)
    projection = lambda_min * e @ e_star
    print(f"projection with limit: ", projection)

    return M - projection