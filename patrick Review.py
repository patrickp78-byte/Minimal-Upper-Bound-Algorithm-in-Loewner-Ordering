# -*- coding: utf-8 -*-
"""
Created on Thu May 15 21:48:02 2025

@author: patrick
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# making elipse
a = np.array([
    [2, 0],
    [0, 1]
])

b = np.array([
    [1, 0],
    [0, 2]
])


c = np.array([
    [2, 0],
    [0, 2]
])

d = np.array([
    [3, np.sqrt(2)],
    [np.sqrt(2), 3]
])


def making_ellipse(matrix):
    # Finding eiganval and eiganvector
    eigans = np.linalg.eig(matrix)
    eiganval = eigans[0]
    eiganve = eigans[1]
    
    # Parameratirization unit circle
    theta = np.linspace(0, 2*np.pi,100)
    x = np.cos(theta)
    y = np.sin(theta)
    creating_circ = np.stack((x,y), axis = 0)
    
    # Transforming circle
    diag = np.diag(1/ np.sqrt(eiganval))
    transform = np.matmul(eiganve,diag)
    ellipse = np.matmul(transform, creating_circ)
    return ellipse


matrix_a = making_ellipse(a)
matrix_b = making_ellipse(b)
matrix_c = making_ellipse(c)
matrix_d = making_ellipse(d)


fig, ax =plt.subplots()
# ax.plot(x,y, label = 'circle')
ax.plot(matrix_a[0,:], matrix_a[1,:], label = 'matrix_a' )
ax.plot(matrix_b[0,:], matrix_b[1,:], label = 'matrix_b' )
ax.plot(matrix_c[0,:], matrix_c[1,:], label = 'matrix_c' )
ax.plot(matrix_d[0,:], matrix_d[1,:], label = 'matrix_d' )
plt.legend(loc= 'upper left')
plt.grid()
plt.show()
sys.exit()

