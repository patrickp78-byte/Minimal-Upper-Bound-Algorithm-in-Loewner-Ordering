# -*- coding: utf-8 -*-
"""
Created on Thu May 15 21:48:02 2025

@author: patrick
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys


dir_path = os.path.dirname(__file__) ; print(dir_path)
current = os.getcwd() ;print(f'old dr: {current}')
os.chdir(dir_path)
new_dr = os.getcwd() ;print(f'new dr: {new_dr}')



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


all_circle = [matrix_a,matrix_b,matrix_c,matrix_d]
names = ['a','b','c','d']

fig, ax =plt.subplots()
for i in range(len(all_circle)):
    # ax.plot(x,y, label = 'circle')
    ax.plot(all_circle[i][0,:], all_circle[i][1,:], label = f'matrix_{names[i]}')
    
plt.legend(loc= 'upper right')
plt.grid()
plt.show()

