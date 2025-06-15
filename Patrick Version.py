
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import os
import sys


def making_ellipse(m,guess):
    """
    Making an Ellipse
    The imputs is a list of matrix
    """
    matrix = m
    matrix.append(guess)
    epsilon = 1e-1
    ellipse_cor = []
    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    for i in matrix:
        # Finding eiganval and eiganvector
        eigans = np.linalg.eig(i)
        eiganval = eigans[0]
        eiganve = eigans[1]
        # Parameratirization unit circle

        theta = np.linspace(0, 2*np.pi,100)
        x = np.cos(theta)
        y = np.sin(theta)
        creating_circ = np.stack((x,y), axis = 0)

        # Transforming circle
        eiganval_zer_check = np.any(eiganval == 0 )
        if eiganval_zer_check == True:
            eiganval = np.where(eiganval > epsilon, eiganval,epsilon)
        else:
            eiganval = eiganval

                  
        diag = np.diag(1/np.sqrt(eiganval))
        transform = np.matmul(eiganve,diag)
        ellipse = np.matmul(transform, creating_circ)
        ellipse_cor.append(ellipse)

    fig, ax =plt.subplots()
    for i in range(len(ellipse_cor)):
    # ax.plot(x,y, label = 'circle')
        ax.plot(ellipse_cor[i][0,:], ellipse_cor[i][1,:], label = f'matrix_{letters[i]}')

    plt.legend(loc = "upper right")
    plt.title(f'{guess}')
    return plt.show()


def null_space(matrix_up):
    u, s, vh = np.linalg.svd(matrix_up)
    rank = (s > 0).sum()
    null_space = vh[rank:].T  # take rows from vh corresponding to zero singular values
    return null_space

 
def is_symmetric(matrix):
    return np.allclose(matrix,matrix.T)

 
def psd_check(matrix):
    """
    PSD and PS Check
    Every PD is PSD
    Positive Definite:
        Symmetric
        All EiganValues are Positive (0 > )
        Determinant are Positive
        Cholestky will work always work
    Positive Semi Defnite:
        Symmetric
        All EiganValues are Positive or Zero (0 >= )
        Determinant = 0 or >0
        Cholestky might fail due to some eiganvalues (diagonals) are zero
    """
    symmetric = is_symmetric(matrix)
    if not is_symmetric(matrix) :
        return False
    is_pos_def = False
    try:
        choletsky = np.linalg.cholesky(matrix)
        is_pos_def = True
    except np.linalg.LinAlgError:
        pass
    return is_pos_def

def eigan_pair_values(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    return eigvals, eigvecs


def new_is_upperbound(m,guess):
    upperbounds= []
    for i in m:
        matrix = guess - i
        if psd_check(matrix) or np.all(np.linalg.eigvalsh(matrix) >= -1e-7):
            upperbounds.append(matrix)

        else:
            return [False , []]
    return [True, upperbounds]



def is_minimal_calc_E(upper,guess):
    """
    This Check if the matrix is minimal by Checking if E = null(C-A) + null(C-B) is span
    if not it will return E = null(C-A) + null(C-B)
    """
    basis = []
    for i in upper:
        null_value = sc.linalg.null_space(i, rcond=1e-7)
        if null_value.size >0:
            basis.append(null_value)
    if not basis:
        return [False, np.zeros((len(upper[0]),0) )]

    E = np.hstack(basis)
    rank = np.linalg.matrix_rank(E)
    return [rank>= len(upper[0]), E]



def old_minimize_upperbound(up_bound,psd_check,guess, E_val):   
    E_perp = null_space(E_val.T)
    
    if E_perp.shape[1] == 0:
        raise RuntimeError("No orthogonal direction found; E_perp is empty.")
    
    eigan_pairs =[ eigan_pair_values(matrix) for matrix in up_bound]
    evals = np.hstack([pair[0] for pair in eigan_pairs ])
    evecs = np.hstack([pair[1] for pair in eigan_pairs ])
    safe_eigs = [ei for ei in evals if ei >= 1e-7]
    # lambda_min = min(safe_eigs)
    
    max_eval_index = np.argmax(evals)
    e = evecs[:, max_eval_index]
    e = e.reshape(-1, 1)
    
    lambdas = []
    for i in range(len(psd_check)):
        if psd_check[i] == "PD":
            # THis grabs the inverse of a matrix for PD
            guess_invers = np.linalg.inv(up_bound[i])
            temp = np.dot((guess_invers@e).reshape(1,-1), e)
            temp[temp == 0] = 1e-7
            alpha = 1/temp
            lambdas.append(alpha)
            
        else: 
            # This grabs the psuido inverse for PSD
            guess_invers = np.linalg.pinv(up_bound[i])
            temp = np.dot((guess_invers@e).reshape(1,-1), e)
            temp[temp == 0] = 1e-7
            alpha = 1/temp
            lambdas.append(alpha)
    
    lambda_min = min(lambdas)
    e_star = e.reshape(1, -1)
    projection = lambda_min * e @ e_star
    new_C = guess - projection

    return new_C


def new_minimize_upperbound(up_bound,guess, E_val):   
    E_perp = null_space(E_val.T)
    
    if E_perp.shape[1] == 0:
        raise RuntimeError("No orthogonal direction found; E_perp is empty.")
    
    rand = np.random.rand(E_perp.shape[1], 1)
    E_perp_rand = E_perp @ rand
    
    
    rand_col_idx = np.random.choice(E_perp_rand.shape[1])
    e = E_perp[:, rand_col_idx].reshape(-1, 1)
    e = E_perp_rand
    
    lambdas = []
    for i in range(len(up_bound)):
        guess_invers = np.linalg.pinv(up_bound[i])
        temp = np.dot(e.reshape(1,-1), guess_invers@e)
        temp[temp == 0] = 1e-7
        alpha = 1/temp
        lambdas.append(alpha)
    
    lambda_min = min(lambdas)
    e_star = e.reshape(1, -1)
    projection = lambda_min * e @ e_star
    new_C = guess - projection

    return new_C




def minimizing_upper_bound(m, guess):
    new_c = [guess]
    # Initial check
    step1, upperbounds = new_is_upperbound(m,new_c[-1])
    
    if step1 is False:
        print(f"{guess} not Upper bound")
        return [guess, "Not Upper Bound"]
    else:
        step2, E = is_minimal_calc_E(upperbounds,new_c[-1])
        
        iteration_count = 0
        
        # if minimal is None or minimal.shape[1] >= len(guess):
        #     raise RuntimeError("Cannot compute projection direction; E spans full space or is None.")
    
        # Loop until is_minimal becomes True
        while not step2:
            # Step 2: this increase the size of the sphere ie makes it bigger to calulate the value
            step3 = new_minimize_upperbound(upperbounds, new_c[-1], E)
            new_c.append(step3)
            
            iteration_count +=1
            # Recheck if the new candidate is minimal
            step1, upperbounds = new_is_upperbound(m,new_c[-1])
            step2, E = is_minimal_calc_E(upperbounds, new_c[-1])
        
        # print(f'{new_c[-1]} is the minimal upper bound')
        # print(f'number of iteration: {iteration_count}')
    
        return [new_c[-1],iteration_count]

def creating_symmetric_matrix(n):
    random = np.random.rand(n,n)
    symmetric = (random @ random.T) 
    return symmetric

def making_upper_bound(A,B):
    
    n = A.shape[0]
    epsilon = 1e-1
    I = np.eye(n)
    
    possible_upper = [np.maximum(A,B)]
    upper_bound_check = new_is_upperbound([A,B],possible_upper[-1])[0]
    
    while not upper_bound_check:
        pos_upper_bound = possible_upper[-1] + epsilon *I
        possible_upper.append(pos_upper_bound)
        upper_bound_check = new_is_upperbound([A,B],possible_upper[-1])[0]
        
        # print(pos_upper_bound)
        # print(upper_bound_check)
    
    return possible_upper[-1]
# sample = creating_symmetric_matrix(10) ;print(sample)

# temp = psd_check(sample); print(temp)




#%%

"""
Testing
"""

n= 2
A = creating_symmetric_matrix(n) #;print(A)
B = creating_symmetric_matrix(n) #;print(B)
C = making_upper_bound(A,B)#;print(C)
making_ellipse([A,B],C)

mew_c, iteration = minimizing_upper_bound([A,B], C)
print(mew_c)
print(iteration)

making_ellipse([A,B,C],mew_c)






# =============================================================================
# step1, upperbounds= new_is_upperbound([A,B],C);print(step1)
# step2, E = is_minimal_calc_E(upperbounds,C);print(step2)
# step3 = new_minimize_upperbound(upperbounds, C, E)
# making_ellipse([A,B,C],step3)
# 
# 
# step1, upperbounds = new_is_upperbound([A,B],step3);print(step1)
# print(upperbounds)
# step2, E = is_minimal_calc_E(upperbounds,C);print(step2)
# step3 = new_minimize_upperbound(upperbounds, C, E)
# 
# 
# step1, upperbounds,psd_check = new_is_upperbound([A,B],step3)
# step2, E = is_minimal_calc_E(upperbounds)
# step3 = new_minimize_upperbound(upperbounds,psd_check, step3, E)

# =============================================================================



























