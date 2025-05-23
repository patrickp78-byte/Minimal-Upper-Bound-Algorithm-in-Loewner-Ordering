# -*- coding: utf-8 -*-
"""
Created on Tue May 21 2025
Last Modified: Tue May 21 2025, 10:30 AM

File Name: graphing_functions.py
Description: Contains functions for generating and plotting ellipses and ellipsoids 
             based on eigenvalues and eigenvectors of matrices. Supports both 2D and 3D plotting.

Notes: 
    - Depends on the 'dependencies' module, which must expose NumPy as 'np' and Matplotlib as 'plt'.
    - Assumes symmetric matrices are provided for valid eigendecomposition.

@author: patrick, jaunie
"""

import dependencies as d

def create_ellipse(eigvals: 'd.np.ndarray', eigvecs: 'd.np.ndarray') -> 'd.np.ndarray':
    """
    Generates a 2D ellipse from eigenvalues and eigenvectors.

    Parameters:
        eigvals : numpy.ndarray
            Array of 2 eigenvalues.
        eigvecs : numpy.ndarray
            Corresponding 2x2 eigenvectors matrix.

    Returns:
        numpy.ndarray
            Transformed points of the ellipse (2, 100)
    """
    # Parameratirization unit circle
    theta = d.np.linspace(0, 2 * d.np.pi, 100)
    unit_circle = d.np.stack((d.np.cos(theta), d.np.sin(theta)), axis=0)
    # Transforming circle
    transform = d.np.matmul(eigvecs, d.np.diag(1 / d.np.sqrt(eigvals)))
    ellipse = d.np.matmul(transform, unit_circle)
    return ellipse


def create_ellipsoid(eigvals: 'd.np.ndarray', eigvecs: 'd.np.ndarray') -> 'd.np.ndarray':
    """
    Generates a 3D ellipsoid from eigenvalues and eigenvectors.

    Parameters:
        eigvals : numpy.ndarray
            Array of 3 eigenvalues.
        eigvecs : numpy.ndarray
            Corresponding 3x3 eigenvectors matrix.

    Returns:
        numpy.ndarray
            Transformed points of the ellipsoid (3, 50, 50)
    """
    axes_length = 1 / d.np.sqrt(eigvals)

    # Create a linear interporlation of phi (from 0 to pi) and theta (from 0 to 2pi) with 50 points
    phi_vals = d.np.linspace(0, d.np.pi, 50)
    theta_vals = d.np.linspace(0, 2 * d.np.pi, 50)
    # This creates a grid of coordinate to create the spear
    # This is done by generating a lattice of all (phi,theta) generated
    phi, theta = d.np.meshgrid(phi_vals, theta_vals)

    # Transform the unit sphere
    unit_sphere = d.np.array([
        d.np.cos(theta) * d.np.sin(phi),
        d.np.sin(theta) * d.np.sin(phi),
        d.np.cos(phi)
    ])

    points = axes_length[:, d.np.newaxis, d.np.newaxis] * unit_sphere
    # apply rotation
    ellipsoid = d.np.matmul(eigvecs, points.reshape(3, -1)).reshape(3, 50, 50)
    return ellipsoid


def plot_ellipse(ellipse_data_list: list[tuple['d.np.ndarray', 'd.np.ndarray']]) -> None:
    """
    Plots one or more ellipses (2D) or ellipsoids (3D) using eigenvalues and eigenvectors.

    Parameters:
        ellipse_data_list : list of tuples
            Each tuple contains:
                - eigvals: numpy.ndarray of eigenvalues (2 or 3)
                - eigvecs: numpy.ndarray of eigenvectors matrix (2x2 or 3x3)

    Returns:
        None

    Raises:
        ValueError
            If eigvals are not 2D or 3D in length.

    Example:
        >>> plot_ellipse([(eigvals2D, eigvecs2D), (eigvals3D, eigvecs3D)])
    """
    has_3d = any(len(eigvals) == 3 for eigvals, _ in ellipse_data_list)

    if has_3d:
        fig = d.plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        d.plt.figure()

    for eigvals, eigvecs in ellipse_data_list:
        if len(eigvals) == 2:
            ellipse = create_ellipse(eigvals, eigvecs)
            d.plt.plot(ellipse[0], ellipse[1])
        elif len(eigvals) == 3:
            ellipsoid = create_ellipsoid(eigvals, eigvecs)
            ax.plot_surface(ellipsoid[0], ellipsoid[1], ellipsoid[2], alpha=0.6)
        else:
            raise ValueError("Eigenvalues must have length 2 or 3 for plotting.")

    if not has_3d:
        d.plt.axhline(0, color='black', linewidth=1)
        d.plt.axvline(0, color='black', linewidth=1)
        d.plt.grid(color='lightgray', linestyle='--')

    d.plt.show()