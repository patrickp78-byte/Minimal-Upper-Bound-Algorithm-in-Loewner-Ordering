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





#%%

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


# This can be better

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

#%%

# Step 2: Define a symmetric positive-definite matrix A (e.g., covariance matrix)
A = np.array([[3, 1, 1],
              [1, 2, 1],
              [1, 1, 1]])

B = np.array([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]
])

C = np.array([
    [6, 2, 1],
    [2, 5, 2],
    [1, 2, 4]
])

D = np.array([
    [6,  2, -1],
    [2,  2,  0],
    [-1, 0,  1]
])

def threeD_graph(matrixz,name):
    
    def three_by_three(matrix):
        #Create a linear interporlation of phi (from 0 to pi) and theta (from 0 to 2pi) with 10 points
        phi_vals = np.linspace(0, np.pi, 20) # UP and down 
        theta_vals = np.linspace(0, 2*np.pi, 20) # Full Rotation around z-axis
        
        # This creates a grid of coordinate to create the spear
        # This is done by generating a lattice of all (phi,theta) generated
        pair_wise = np.meshgrid(phi_vals, theta_vals)
        
        # This separates the lattice points into phi and theta
        phi= pair_wise[0] #(Nort to South)
        theta = pair_wise[1] #(Rotations)
        
        # Creating points to plot in a 3D
        r1 = 1 # centered at this point
        x = r1* np.sin(phi) * np.cos(theta)
        y = r1* np.sin(phi) * np.sin(theta)
        z = r1* np.cos(phi)
        
        sphere = np.vstack((x.flatten(),y.flatten(),z.flatten())) # a faster way to do this is by using np.ravel()
        
        
        # Rotating using eigenvalues and vectors
        eigans = np.linalg.eig(matrix) # Calculates the eigen values and vectors
        eigan_val =  eigans[0]
        eigan_vec =  eigans[1]
        
        # Scaling 
        scal = np.sqrt(eigan_val)
        diagonalization = np.diag(scal)
        # Transforming spear into an ellipse using eiganvalues and eigan vectors of matrix
        tranform = np.matmul(eigan_vec,diagonalization)
        
        # transformed spear
        new_sphere = np.matmul(tranform, sphere)
        # New sphear coordinate
        x_trans = new_sphere[0,:].reshape(x.shape)
        y_trans = new_sphere[1,:].reshape(y.shape)
        z_trans = new_sphere[2,:].reshape(z.shape)
        
        return [x_trans,y_trans,z_trans]

    m_name = name
    color_names = ['blue', 'orange', 'green', 'red', 'purple',
               'brown', 'pink', 'gray', 'olive', 'cyan']
    alpha = np.linspace(0, 0.5, len(matrixz))
    matrix_x = []

    for i in matrixz:
        coordinate = three_by_three(i)
        matrix_x.append(coordinate)
    
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1,1,1, projection='3d')
    for i in range(len(matrix_x)):
        new_coor = matrix_x[i]
        x,y,z = new_coor[0],new_coor[1],new_coor[2]   
        #ax.plot_surface(x,y,z, color=colors[i][0],  edgecolor = colors[i][1],  alpha = colors[i][2], label = f'matrix: {colors[i][3]}' )
        ax.plot_surface(x,y,z, edgecolor = color_names[i],  alpha = np.round(alpha[i],decimals= 2), label = f'matrix: {m_name[i]}' )
    ax.set_box_aspect([1, 1, 1])
    ax.set_title('Ellipse')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')  
    plt.legend()


name= ["A","B","C","D"]
temp = threeD_graph([A,B,C,D],name)


#%%
import scipy as sc


# Checks if matrix C is a symmetrix and minimal for A and C

A = np.array([[1,0],
              [0,0]])

B = np.array([[0,0],
              [0,1]])

# C =np.array([[1,0],
#              [0,1]])


C =np.array([[2,np.sqrt(2)],
              [np.sqrt(2),2]])

# A = np.array([
#     [2, -1,  0],
#     [-1, 2, -1],
#     [0, -1,  2]
# ])

# B = np.array([
#     [1, 2, 3],
#     [0, 4, 5],
#     [0, 0, 6]
# ])

def minimal_check(a,b,c):
    ma_array = [a,b,c]
    # Checks if a matrix Semi Positive Matrix
    def psd_check(matrix):
        
        symmetric = np.allclose(matrix, matrix.T)
        if symmetric == False:
            return f'Matrix Not Symmetric'
        else:
            try:
                chol = np.linalg.cholesky(matrix)
                return f'Positive Semi Definite'
            except np.linalg.LinAlgError as e:
                return f'Positive Definite'
    check = []
    for i in ma_array:
        checking = psd_check(i)
        if checking == 'Positive Semi Definite' or checking =='Positive Definite':
            check.append(True)
        else:
            check.append(False)
            
    is_true = all(check)
    if is_true == True:
        c_minus_a = ma_array[2] -ma_array[0]
        c_minus_b = ma_array[2] -ma_array[1]      
        c_minus_a_null = sc.linalg.null_space(c_minus_a)
        c_minus_b_null = sc.linalg.null_space(c_minus_b)  
        # combines the 2 matrix together
        combine_nulls = np.column_stack([c_minus_a_null,c_minus_b_null])
        # uses SVD to count how many linearly independet col we have
        rank = np.linalg.matrix_rank(combine_nulls)
        # This tells us how many columns we have in the matrix
        num_cols = combine_nulls.shape[1]
        if rank == num_cols:
            return print(f'Matrix C is minimal: {c}')
        else:
            return print(f'Matrix C is minimal: {c}')
    else:
        print('one of the matrix is not Symmetric')

minimality_check = minimal_check(A,B,C)




sys.exit()


#scipy method
null = sc.linalg.null_space(C - B)

tol = 1e-10
u, s, vh = np.linalg.svd(C-B)
rank = (s>tol).sum()
null_space = vh[rank:].T














