"""
Implementation of test functions (21) through (31) and their derivatives 
(gradient vectors and Hessian matrices) from More et al., "Testing 
Unconstrained Optimization Software" (1981, Argonne National Laboratory), and 
select test functions used in "Testing Parallel Variable Transformation" 
(Yamakawa and Fukushima, 1999). 

Unless explicitly said the functions are identified by their number in 
the article, and are of the form

f(x) = sum_{i=1}^m f_i(x)**2

where x is an n-dimensional array. The n-dimensional gradients are of the form

grad(f) = 2 * sum_{i=1}^m [ f_i(x) * grad(f_i) ]

and n-by-n Hessian matrices are given by

Hess(f) = 2 * sum_{i=1}^m [ f_i(x) * Hess(f_i) + grad(f_i) grad(f_i)^T ].

The function starting_x contains the suggested optimization starting points 
for each of the mathematical functions. 

(1)
*_d functions have the following inputs and outputs:
    
    Args:
        x (ndarray): Point at which function evaluations are made.
        out (str): Chooses the output - {'Grad' for gradient (default)}
                                        {'Hess' for Hessian matrix}
    Returns: 
        {if out='Grad'} g (ndarray): Gradient of f at x.
        {if out='Hess'} H (ndarray): Hessian matrix of f at x.
        
(2)
Uppercase letters refers to matrices. Specifically, A is used to denote the
``gradient matrix'', i.e. the jth column of A is the gradient of f_j, 
1 <= j <= m.

(3)
Functions below with sparse Hessians make use of scipy sparse matrices. 
Hessians are returned as scipy.sparse.csc matrices.
rosen_ext, powell_ext, broyden_tri, shanno_quad, arrowhead, nondiag_quartic

"""

import numpy as np

from scipy import sparse

def starting_x(f, n):
    """Returns starting point for optimization scheme as given in the paper.
    
    Args:
        f (func): Function whose starting point we wish to obtain.
        n (int): Number of dimensions.
    Returns: 
        x (ndarray): n-dimensional vector
    """

    if   (f.__name__ == 'rosen_ext'):
        x = np.zeros(n)
        x[::2] = -1.2
        x[1::2] = 1
        return x

    elif (f.__name__ == 'powell_ext'):
        x = np.zeros(n)
        x[::4]  = 3
        x[1::4] = -1
        x[2::4] = 0
        x[3::4] = 1
        return x.astype(float)

    elif (f.__name__ == 'pen_func1'):
        return np.arange(1, n+1).astype(float)

    elif (f.__name__ == 'pen_func2' or f.__name__ == 'brown'):
        return 0.5 * np.ones(n)

    elif (f.__name__ == 'vardim_func'):
        return (1 - np.arange(1, n+1) / n)

    elif (f.__name__ == 'trig_func'):
        return np.ones(n) / n

    elif (f.__name__ == 'disc_bv_func' or f.__name__ == 'disc_int_func'):
        j = np.arange(1, n+1)
        return np.multiply(j, j / (n+1) - 1) / (n+1)

    elif (f.__name__ == 'broyden_tri' or f.__name__ == 'broyden_band'
          or f.__name__ == 'shanno_quad'):
        return -np.ones(n)

    elif (f.__name__ == 'nondiag_quartic'):
        return np.array([(-1)**i for i in range(n)])

    elif (f.__name__ == 'arrowhead'):
        return np.ones(n)

    else:
        raise ValueError('Starting x for this function could not be found.')


def rosen_ext(x):
    """(21) Extended Rosenbrock function of n (even) variables with m=n:
    
    f_{2i-1} = 10*(x_{2i} - x_{2i-1}**2)
    f_{2i} = (1 - x_{2i-1})
    """
    if(np.size(x) % 2 != 0):
        raise ValueError("This function requires an even number of "
                         "dimensions.")
    n = np.size(x)
    f = np.zeros(n)
    f[::2] = 10 * (x[1::2] - x[::2]**2)
    f[1::2] = 1 - x[::2]
    
    return np.sum(f**2)

def rosen_ext_d(x, out='Grad'):
    """Gradient/Hessian of the extended Rosenbrock function (21) of n (even) 
    variables.    
    """
    n = np.size(x) 
    f = np.zeros(n)
    f[::2] = 10 * (x[1::2] - x[::2]**2)
    f[1::2] = 1 - x[::2]
    matrix_data = {}
    
    for i in range(n):
        if ((i+1) % 2 == 1):
            matrix_data[(i, i)] = -20 * x[i]
            matrix_data[(i+1, i)] = 10
            
        else:
            matrix_data[(i-1, i)] = -1
    
    row_ind = [item[0] for item in matrix_data.keys()]
    col_ind = [item[1] for item in matrix_data.keys()]
    data = [val for val in matrix_data.values()]
    
    A_row = sparse.csr_matrix((data, (row_ind, col_ind)))
    
    if (out == 'Grad'):
        return 2 * A_row.dot(f)

    elif (out == 'Hess'):
        # Save as csc_matrix to facilitate column slicing.
        A_col = A_row.tocsc()
        H = sparse.lil_matrix((n, n))
        
        for i in range(n):
            # Only nonzero second derivatives come from f_{2i-1} terms.
            if ((i+1) % 2 == 1):
                H[i, i] += -20 * f[i]
        
        H = H.tocsc()
        for i in range(n):
            H += A_col[:, i].dot(A_col[:, i].transpose())
                
        return 2 * H
    else:
        raise ValueError("Wrong function inputs.")


def powell_ext(x):
    """(22) Extended Powell function of n (visible by 4) variables with m=n:
    
    f_{4i-3} = x_{4i-3} + 10*x_{4i-2}
    f_{4i-2} = sqrt(5)*(x_{4i-1} - x_{4i})
    f_{4i-1} = (x_{4i-2} - 2*x_{4i-1})**2
    f_{4i} = sqrt(10)*(x_{4i-1} - x_{4i})**2
    """
    
    if(np.size(x) % 4 != 0):
        raise ValueError("This function requires n (dimension number "
                         "divisible by 4.")
    
    f1 = x[::4] + 10 * x[1::4]
    f2 = np.sqrt(5) * (x[2::4] - x[3::4])
    f3 = (x[1::4] - 2 * x[2::4])**2
    f4 = np.sqrt(10) * (x[::4] - x[3::4])**2
    
    return np.sum(f1**2 + f2**2 + f3**2 + f4**2)

def powell_ext_d(x, out='Grad'):
    """Gradient/Hessian of the extended Powell function (22) of n 
    (visible by 4) variables.
    """
    
    if(np.size(x) % 4 != 0):
        raise ValueError("This function requires n (dimension number "
                         "divisible by 4.")
    
    n = np.size(x) 
    f = np.zeros(n)
    matrix_data = {}
    
    for i in range(n):
        if ((i+1) % 4 == 1):
            f[i] = x[i] + 10 * x[i+1]
            matrix_data[(i, i)] = 1
            matrix_data[(i+1, i)] = 10
            
        elif ((i+1) % 4 == 2):
            f[i] = np.sqrt(5) * (x[i+1] - x[i+2])
            matrix_data[(i+1, i)] = np.sqrt(5)
            matrix_data[(i+2, i)] = -np.sqrt(5)
          
        elif ((i+1) % 4 == 3):
            f[i] = (x[i-1] - 2 * x[i])**2
            matrix_data[(i-1, i)] = 2 * (x[i-1] - 2 * x[i])
            matrix_data[(i, i)] = -4 * (x[i-1] - 2 * x[i]) 
         
        elif ((i+1) % 4 == 0):
            f[i] = np.sqrt(10) * (x[i-3] - x[i])**2
            matrix_data[(i-3, i)] = 2 * np.sqrt(10) * (x[i-3] - x[i])
            matrix_data[(i, i)] = -2 * np.sqrt(10) * (x[i-3] - x[i])
    
    row_ind = [item[0] for item in matrix_data.keys()]
    col_ind = [item[1] for item in matrix_data.keys()]
    data = [val for val in matrix_data.values()]
    
    A_row = sparse.csr_matrix((data, (row_ind, col_ind)))
    
    if (out == 'Grad'):
        return 2 * A_row.dot(f)
    
    elif (out == 'Hess'):
        # Save as csc_matrix to facilitate column slicing.
        A_col = A_row.tocsc()
        H = sparse.lil_matrix((n, n))
        
        # Only nonzero second derivatives come from f_{4i-1} and f_{4i} terms.
        # The values are reflected in the two 4x4 blocks below. 
        
        block3 = sparse.csc_matrix(np.array([[0, 0, 0, 0], [0, 2, -4, 0], [0, -4, 8, 0], 
                           [0, 0, 0, 0]]))
        block4 = sparse.csc_matrix(np.array([[2*np.sqrt(10), 0, 0, -2*np.sqrt(10)], 
                           [0, 0, 0, 0], [0, 0, 0, 0], 
                           [-2*np.sqrt(10), 0, 0, 2*np.sqrt(10)]]))

        for i in range(int(n/4)):
            H[4*i:4*(i+1), 4*i:4*(i+1)] += f[4*i + 2] * block3 + \
                                           f[4*i + 3] * block4
                         
        H = H.tocsc()
        for i in range(n):
            H += A_col[:, i].dot(A_col[:, i].transpose())
                            
        return 2 * H
    else:
        raise ValueError("Wrong function inputs.")


def pen_func1(x):
    """(23) Penalty function I of variable n with m = n+1:
    
    f_i = sqrt(10^{-5})*(x_i - 1), 1 <= i <= n
    f_{n+1} = (sum_i^n x_j**2) - 1/4
    """

    f_np1 = np.sum(x**2) - 0.25
    #min([f_np1**2, 10**20])
    return 10**(-5) * np.sum((x - 1)**2) + f_np1**2

def pen_func1_d(x, out='Grad'):
    """Gradient/Hessian of the penalty function I (23) of variable n."""
    
    n = np.size(x)
    
    fi = np.sqrt(10**-5) * (x-1)
    #f_np1 = min([np.sum(x**2) - 0.25, 1e10])
    f_np1 = np.sum(x**2) - 0.25
    
    A = np.zeros((n, n + 1))
    np.fill_diagonal(A, np.sqrt(10**-5))
    A[:, -1] = 2 * x
    
    if(out == 'Grad'):
        return 2 * np.add(np.dot(A[:, :-1], fi), A[:, -1] * f_np1)
   
    elif(out =='Hess'):
        H = np.zeros((n, n))
        
        for i in range(A.shape[1]):
            H += np.matmul(A[:, i].reshape(n, 1), A[:, i].reshape(1, n))
        
        # Only nonzero second derivatives come from i = n+1 term and 
        # correspond to the diagonal elements of Hess(f_{n+1}), which are 2 
        H += f_np1 * 2 * np.identity(n)
        return 2 * H
    else:
        raise ValueError("Wrong function inputs.")


def pen_func2(x):
    """(24) Penalty function II of variable n with m = 2n:
    
    f1 = x1 - 0.2
    f_i = sqrt(10^{-5})*(exp(x_i/10) + exp(x_{i-1}/10) - y_i), 2 <= i <= n
    f_i = sqrt(10^{-5})*(exp(x_{i-n+1}/10) - exp(-1/10) - y_i), n < i < 2n
    f_{2n} = (sum_j^n (n-j+1)*x_j^2) - 1
    
    where
    
    y_i = exp(i/10) + exp((i-1)/10)
    """

    n = np.size(x)
    f = np.zeros(2*n)
    yi = np.exp(np.arange(2, n+1) / 10) + np.exp(np.arange(1, n) / 10) 
    
    f[0] = x[0] - 0.2
    f[1:n] = np.sqrt(10**-5)*(np.exp(x[1:] / 10) + np.exp(x[:-1] / 10) - yi)
    f[n:-1] = np.sqrt(10**-5) * (np.exp(x[1:] / 10) - np.exp(-1 / 10))
    f[-1] = np.sum((n - np.arange(1, n+1) + 1) * x**2) - 1
    
    return np.sum(f**2)

def pen_func2_d(x, out='Grad'):
    """Gradient/Hessian of penalty function II (24) of variable n."""   
    
    n = np.size(x)
    f = np.zeros(2*n)
    yi = np.exp(np.arange(2, n+1) / 10) + np.exp(np.arange(1, n) / 10) 
    
    f[0] = x[0] - 0.2
    f[1:n] = np.sqrt(10**-5) * (np.exp(x[1:] / 10) + np.exp(x[:-1] / 10) - yi)
    f[n:-1] = np.sqrt(10**-5) * (np.exp(x[1:] / 10) - np.exp(-1 / 10))
    f[-1] = np.sum((n - np.arange(1, n+1) + 1) * x**2) - 1
    
    A = np.zeros((n, 2*n))
    
    # Contribution from f_1
    A[0, 0] = 1 
    
    for i in range(1, n):
        A[:, i] += 0.1 * np.sqrt(10**-5) * np.exp(x[i]/10) 
        A[:, i-1] += 0.1 * np.sqrt(10**-5) * np.exp(x[i-1]/10)
        A[:, i+n-1] += 0.1 * np.sqrt(10**-5) * np.exp(x[i]/10)
        
    A[:, -1] += 2 * np.multiply(x, n + 1 - np.arange(1, n+1))
    
    if(out == 'Grad'):
        g = np.matmul(A, f)
        return 2 * g
 
    elif(out =='Hess'): 
        H = np.zeros((n, n))
        
        for i in range(A.shape[1]):
            H += np.matmul(A[:, i].reshape(n, 1), A[:, i].reshape(1, n))
        
        for i in range(1, n):
            # 2 <= i <= n
            H[i, i] += f[i] * 0.01 * np.sqrt(10**-5) * np.exp(x[i]/10) 
            H[i-1, i-1] += f[i] * 0.01 * np.sqrt(10**-5) * np.exp(x[i-1]/10) 
            # n < i < 2n
            H[i, i] += f[i+n-1] * 0.01 * np.sqrt(10**-5) * np.exp(x[i]/10)
        
        # i = 2n
        H += 2 * f[-1] * (n + 1 - np.arange(1, n+1)) * np.eye(n)
        return 2 * H
    else:
        raise ValueError("Wrong function inputs.")


def vardim_func(x):
    """(25) Variably dimensioned function of variable n with m = n+2:
        
    fi = xi - 1
    f_{n+1} = sum_j^n j*(x_j - 1)
    f_{n+2} = (sum_j^n j*(x_j - 1))^2
    """
    
    n = np.size(x) 
    
    fi = x - 1
    f_np1 = np.sum(np.multiply(np.arange(1, n+1), x-1))

    return np.sum(fi**2) + f_np1**2 + f_np1**4

def vardim_func_d(x, out='Grad'):
    """Gradient/Hessian of the variably dimensioned function (25) of 
    variable n."""
    
    n = np.size(x) 
    f = np.zeros(n+2)
    
    f[:-2] = x-1
    f[-2] = np.sum(np.multiply(np.arange(1, n+1), x-1))
    f[-1] = f[-2]**2
    
    A = np.zeros((n, n+2))
    np.fill_diagonal(A, np.ones(n))
    
    # Contribution from f_{n+1}
    A[:, -2] = np.arange(1, n+1)
    
    # Contribution from f_{n+2}
    A[:, -1] = 2 * np.arange(1, n+1) * f[-2]
    
    if(out == 'Grad'):
        g = np.matmul(A, f)
        return 2 * g
 
    elif(out =='Hess'): 
        H = np.zeros((n, n))
        
        for i in range(A.shape[1]):
            H += np.matmul(A[:, i].reshape(n, 1), A[:, i].reshape(1, n))
        
        # Only nonzero second derivatives come from f_{n+2}
        Hnp2 = np.arange(1, n+1)
        H += f[-1] * 2 * np.matmul(Hnp2.reshape(n, 1), Hnp2.reshape(1, n))
        return 2 * H
    else:
        raise ValueError("Wrong function inputs.")
    
    
def trig_func(x):
    """(26) Trigonometric function of variable n with m = n:
    
    f_i = n - sum_j^n cos(x_j) + i*(1 - cos(x_i)) - sin(x_i), 1 <= i < n
    f_n = (product_j^n x_j) - 1
    """
    
    n = np.size(x) 
    
    fi = n - np.sum(np.cos(x)) + \
          np.multiply(np.arange(1, n+1), 1 - np.cos(x)) - np.sin(x)   
    
    return np.sum(fi**2) 

def trig_func_d(x, out='Grad'):
    """Gradient/Hessian of the trigonometric function (26) of variable n."""
    
    n = np.size(x) 
    f = np.zeros(n)
    for i in range(n):
        f[i] = n - np.sum(np.cos(x)) + (i+1) * (1 - np.cos(x[i])) \
                  - np.sin(x[i])  
    
    A = np.zeros((n, n))
    gii = np.multiply(np.arange(1, n+1), np.sin(x)) - np.cos(x)
    np.fill_diagonal(A, gii)
    
    # Adds same vector to all columns of A. Accounts for contribution to the
    # full gradient of d/dx_i f_i = sin(x_i) (same form for all i).
    A += np.tile(np.sin(x), (n, 1)).transpose()
    
    if(out == 'Grad'):
        g = np.matmul(A, f)
        return 2 * g
 
    elif(out =='Hess'): # this doesnt work yet - diagonal terms are off
        H = np.zeros((n, n))
        j = np.arange(n)
        for i in range(n):
            H += np.matmul(A[:, i].reshape(n, 1), A[:, i].reshape(1, n))
            
            Hi = np.zeros((n, n))
            # Only nonzero second derivatives are the diagonal 
            # terms of Hess(f_i).
            Hi[i, i] = ((i + 2) * np.cos(x[i]) + np.sin(x[i]))
            cond = (j != i)
            Hi[cond, cond] = np.cos(x[cond])
            
            H += f[i] * Hi
        return 2 * H
    else:
        raise ValueError("Wrong function inputs.")


def brown(x):
    """(27) Brown almost-linear function of variable n with m=n:
        
    f_i = x_i + sum_j^n x_j - (n+1), 1 <= i < n
    f_n = (product_j^n x_j) - 1   
    """

    fi = x[:-1] + np.sum(x) - np.size(x) - 1 
    fn = np.prod(x) - 1
    return np.sum(fi**2) + fn**2

def brown_d(x, out='Grad'):
    """Gradient/Hessian of the Brown almost-linear function (27) of 
    variable n."""

    n = np.size(x)
    f = np.zeros(n)
    f[:-1] = x[:-1] + np.sum(x) - (n + 1)
    f[-1] = np.prod(x) - 1
    
    # Initialize A as ones instead of zeros (since d/dx_j f_i = 1, i!=j)
    A = np.ones((n, n))
    np.fill_diagonal(A, 2)
    
    for i in range(n):
        A[i, -1] = np.prod(np.delete(x, i))

    if(out == 'Grad'):
        g = np.matmul(A, f)
        return 2 * g
 
    elif(out =='Hess'): 
        H = np.zeros((n, n))
        
        for i in range(n):
            H += np.matmul(A[:, i].reshape(n, 1), A[:, i].reshape(1, n))
        
        Hi = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if (i != j):
                    Hi[i, j] += np.prod(np.delete(x, [i, j]))
        H += f[-1] * Hi
        return 2 * H
    else:
        raise ValueError("Wrong function inputs.")


def disc_bv_func(x):
    """(28) Discrete boundary value function of variable n with m=n:
        
    f_i = 2x_i - x_{i-1} - x_{i+1} + 0.5*h^2*(x_i + t_i + 1)^3 
    
    where
    
    h = 1/(n+1)
    t_i = i*h
    x_0 = x_{n+1} = 0
    """

    x_pad = np.pad(x, 1, 'constant', constant_values=0)
    n = np.size(x)
    h = 1 / (n + 1)
    fi = 2 * x - x_pad[:-2] - x_pad[2:] + \
         0.5 * h**2 * (x + h * np.arange(1, n+1) + 1)**3

    return np.sum(fi**2)

def disc_bv_func_d(x, out='Grad'):
    """Gradient/Hessian of the discrete boundary value function (28) of 
    variable n."""

    # Add one zero to each end of 'x' array 
    x_pad = np.pad(x, 1, 'constant', constant_values=0)

    n = np.size(x)
    h = 1 / (n + 1)
    t = h * np.arange(1, n+1)
    f = np.zeros(n)
    for i in range(n):
        f[i] = 2 * x[i] - x_pad[:-2][i] - x_pad[2:][i] + \
               0.5 * h**2 * (x[i] + t[i] + 1)**3
    
    A = np.zeros((n, n))
    diagonal = np.ones(n)
    
    A += np.diag(diagonal * (2 + (3/2) * h**2 * (x + t + 1)**2), 0)
    A -= np.diag(diagonal[:-1], 1)
    A -= np.diag(diagonal[:-1], -1)
    
    if (out == 'Grad'):
        g = np.matmul(A, f)
        return 2 * g
        
    elif (out == 'Hess'): 
        H = np.zeros((n, n))
        
        for i in range(n):
            H += np.matmul(A[:, i].reshape(n, 1), A[:, i].reshape(1, n))
        
        # Only nonzero second derivatives are the diagonal terms of Hess(f_i)
        H += 3 * h**2 * np.multiply(f, x + t + 1) * np.eye(n)
        return 2 * H
    else:
        raise ValueError("Wrong function inputs.")
    

def disc_int_func(x):    
    """(29) Discrete integral equation function of variable n with m=n:
        
    f_i = x_i + 0.5*h*[(1-t_i) * sum_j^i t_j * (x_j + t_j + 1)^3
                   + t_i * sum_{j=i+1}^n (1-t_j)*(x_j + t_j + 1)^3]
    where
    
    h = 1/(n+1)
    t_i = i*h
    x_0 = x_{n+1} = 0
    """
    
    n = np.size(x)
    h = 1 / (n + 1)
    t = h * np.arange(1, n+1)
    f = np.zeros(n)
    
    for i in range(n):
        s1 = np.sum(np.multiply(t[:i+1], (x[:i+1] + t[:i+1] + 1)**3))
        s2 = np.sum(np.multiply((1-t[i+1:]), (x[i+1:] + t[i+1:] + 1)**3))
        f[i] = x[i] + 0.5 * h * ((1 - t[i]) * s1 + t[i] * s2)

    return np.sum(f**2)

def disc_int_func_d(x, out='Grad'):    
    """Gradient/Hessian of the discrete integral equation function (29) of 
    variable n."""
    
    n = np.size(x)
    h = 1 / (n + 1)
    t = h * np.arange(1, n+1)
    f = np.zeros(n)
    A = np.zeros((n, n))
    np.fill_diagonal(A, np.ones(n))
    g = np.zeros(n)
    
    for i in range(n):
        s1 = np.sum(np.multiply(t[:i+1], (x[:i+1] + t[:i+1] + 1)**3))
        s2 = np.sum(np.multiply((1-t[i+1:]), (x[i+1:] + t[i+1:] + 1)**3))
        f[i] = x[i] + 0.5 * h * ((1 - t[i]) * s1 + t[i] * s2)
        
        A[i, i-n:] += 1.5 * h * (1 - t[i:]) * t[i] * (x[i] + t[i] + 1)**2
        A[i, :i-n] += 1.5 * h * t[:i-n] * (1 - t[i]) * (x[i] + t[i] + 1)**2
    
    if (out == 'Grad'):
        g = np.matmul(A, f)
        return 2 * g
        
    elif (out == 'Hess'): 
        H = np.zeros((n, n))
        
        for i in range(n):
            H += np.matmul(A[:, i].reshape(n, 1), A[:, i].reshape(1, n))
            
            # Only nonzero second derivatives are the diagonal 
            # terms of Hess(f_i).
            Hi = np.zeros((n, n))
            j1 = np.arange(i+1)
            j2 = np.arange(i, n)
            
            Hi[j1, j1] = 3 * h * (1 - t[i]) * t[j1] * (x[j1] + t[j1] + 1)
            Hi[j2, j2] = 3 * h * t[i] * (1 - t[j2]) * (x[j2] + t[j2] + 1)         
            H += Hi * f[i]
        return 2 * H
    else:
        raise ValueError("Wrong function inputs.")


def broyden_tri(x):
    """(30) Broyden tridiagonal function of variable n with m=n:
        
    f_i =(3-2x_i)x_i - x_{i-1} - 2x_{i+1} + 1
    
    where
    
    x_0 = x_{n+1} = 0
    """

    x_pad = np.pad(x, 1, 'constant', constant_values=0)
    f = np.multiply(3 - 2 * x, x) - x_pad[:-2] - 2 * x_pad[2:] + 1
    
    return np.sum(f**2)

def broyden_tri_d(x, out='Grad'):
    """Gradient/Hessian of the broyden tridiagonal function (30) of 
    variable n."""

    n = np.size(x)
    x_pad = np.pad(x, 1, 'constant', constant_values=0)
    f = np.multiply(3 - 2 * x, x) - x_pad[:-2] - 2 * x_pad[2:] + 1

    A = sparse.lil_matrix((n, n))
    A.setdiag(3 - 4 * x)
    A.setdiag(-1, k=1)
    A.setdiag(-2, k=-1)

    if (out == 'Grad'):
        A_row = A.tocsr()
        return 2 * A_row.dot(f)
        
    elif (out == 'Hess'): 
        H = sparse.csc_matrix((n, n))
        A_col = A.tocsc()
        
        for i in range(n):
            H += A_col[:, i].dot(A_col[:, i].transpose())
            H[i, i] += -4 * f[i]
                
        return 2 * H
    else:
        raise ValueError("Wrong function inputs.")


def broyden_band(x):
    """(31) Broyden banded function of variable n with m=n:
        
    f_i =(2+2x_i^2)x_i + 1 - sum_{j in J} x_j*(1+x_j)
    
    where
        
    J = {j: j!=i and max(1,i-5) <= j <= min(n, i+1)}
    """
    n = np.size(x)
    f = np.multiply(x, 2 + 5 * x**2) + 1
    j = np.arange(1,n+1)
    
    for i in range(1, n+1):
        ind = (j >= max(1, i-5)) & (j <= min(n, i+1)) & (j != i)
        f[i-1] -= np.sum(x[ind] * (1 + x[ind]))
        
    return np.sum(f**2)

def broyden_band_d(x, out='Grad'):
    """Gradient/Hessian of the broyden banded function (31) of variable n."""

    n = np.size(x)
    f = np.multiply(x, 2 + 5 * x**2) + 1
    A = np.zeros((n, n))
    diagonal = np.ones(n)
    A += np.diag(diagonal * (2 + 15 * x**2), 0)
    j = np.arange(1,n+1)
    
    for i in range(1, n+1):
        ind = (j >= max(1, i-5)) & (j <= min(n, i+1)) & (j != i)
        f[i-1] -= np.sum(x[ind] * (1 + x[ind]))
        A[ind, i-1] -= (1 + 2 * x[ind])

    if (out == 'Grad'):
        g = np.matmul(A, f)
        return 2 * g
        
    elif (out == 'Hess'): 
        H = np.zeros((n, n))
        
        for i in range(n):
            H += np.matmul(A[:, i].reshape(n, 1), A[:, i].reshape(1, n))
            Hi = np.zeros((n,n))
            H[i, i] += 30 * x[i] * f[i]
            
            j = np.arange(1,n+1)
            ind = (j >= max(1, i+1-5)) & (j <= min(n, i+1+1)) & (j != i+1)
            
            Hi[ind, ind] = -2
            H += f[i] * Hi
        return 2 * H
    else:
        raise ValueError("Wrong function inputs.")


def shanno_quad(x):
    """Shanno’s quadratic tridiagonal function of variable n.
    Problem 6 from Fukushima (1999):
        
        f1 = 2x_1 - 1
        f_i = sqrt(i) * (2x_{i-1} + x_i), (1 < i <= n)
    """
    if (np.size(x) < 2):
        raise ValueError('This function requires n > 2.')
    
    n = np.size(x)
    f1 = 2 * x[0] - 1
    fn = np.sqrt(np.arange(2, n+1)) * (2 * x[:-1] - x[1:]) 
    
    return f1**2 + np.sum(fn**2)

def shanno_quad_d(x, out='Grad'):
    """Gradient of the Shanno’s quadratic tridiagonal function of variable n.
    Problem 6 from Fukushima (1999).
    """
    
    if (np.size(x) < 2):
        raise ValueError('This function requires n > 1.')
        
    n = np.size(x)
    f = np.zeros(n)
    f[0] = 2 * x[0] - 1
    f[1:] = np.sqrt(np.arange(2, n+1)) * (2 * x[:-1] - x[1:]) 
    
    A = sparse.lil_matrix((n, n))
    
    A[0, 0] += 2
    for i in range(n-1):
        A[i, i+1] += np.sqrt(i+2) * 2 
        A[i+1, i+1] += -np.sqrt(i+2)
    
    if (out == 'Grad'):
        A_row = A.tocsr()
        return 2 * A_row.dot(f)
        
    elif (out == 'Hess'): 
        H = sparse.csc_matrix((n, n))
        A_col = A.tocsc()
        
        for i in range(n):
            H += A_col[:, i].dot(A_col[:, i].transpose())
            
        return 2 * H
    else:
        raise ValueError("Wrong function inputs.")


def arrowhead(x):
    """Arrowhead function of variable n.
    Problem 13 from Fukushima (1999).
    """
    if (np.size(x) < 2):
        raise ValueError("This function requires n > 1.")

    return np.sum((x[:-1]**2 + x[-1]**2)**2 - 4 * x[:-1] + 3)

def arrowhead_d(x, out='Grad'):
    """Gradient of the arrowhead function of variable n.
    Problem 13 from Fukushima (1999).
    """
    
    if (np.size(x) < 2):
        raise ValueError("This function requires n > 1.")
        
    n = np.size(x)
    g = np.zeros(n)
    g[:-1] = 4 * (x[:-1] * (x[:-1]**2 + x[-1]**2) - 1)
    g[-1] = 4 * np.sum(x[-1] * (x[:-1]**2 + x[-1]**2))
    
    if (out == 'Grad'):
        return g
        
    elif (out == 'Hess'): 
        H = sparse.lil_matrix((n, n))
        H[:-1, :-1] += np.diag(4 * (3 * x[:-1]**2 + x[-1]**2))
        H[-1, :-1] += 8 * (x[:-1] * x[-1]).reshape(1, -1)
        H[:-1, -1] += 8 * (x[:-1] * x[-1]).reshape(-1, 1)
        H[-1, -1] += 4 * np.sum(x[:-1]**2 + 3 * x[-1]**2)
        return H.tocsr()
    
    else:
        raise ValueError("Wrong function inputs.")


def nondiag_quartic(x):
    """Nondiagonal quartic function of variable n.
    Problem 14 from Fukushima (1999):
        
        f1 = x_1 - x_2
        f_i = (x_{i-1} + x_i + x_n)^2, (1 < i < n)
        fn = x_{n-1} - x_n
    """
    if (np.size(x) < 3):
        raise ValueError("This function requires n > 2.")
    f1 = (x[0] - x[1])
    fn = (x[-2] - x[-1])

    return f1**2 + fn**2 + np.sum((x[:-2] + x[1:-1] + x[-1])**4)

def nondiag_quartic_d(x, out='Grad'):
    """Gradient of the nondiagonal quartic function of variable n.
    Problem 14 from Fukushima (1999).
    """
    
    if (np.size(x) < 3):
        raise ValueError("This function requires n > 2.")
        
    n = np.size(x)
    f = np.zeros(n)
    f[0] = (x[0] - x[1])
    f[-1] = (x[-2] - x[-1])
    f[1:-1] = (x[:-2] + x[1:-1] + x[-1])**2
    
    vec = 2 * (x[:-2] + x[1:-1] + x[-1])
    
    A = sparse.lil_matrix((n, n))
    A[0, 0] = 1
    A[1, 0] = -1
    A[-1, -1] = -1
    A[-2, -1] = 1
    
    for i in range(n-2):
        A[i+1, i+1] += vec[i]
        A[i, i+1] += vec[i]
        A[-1, i+1] += vec[i]

    if (out == 'Grad'):
        A = A.tocsr()
        return 2 * A.dot(f)
        
    elif (out == 'Hess'): 
        A_col = A.tocsc()
        H = sparse.lil_matrix((n, n))
        
        for i in range(n):
            if (i < n-2):
                fi = f[i+1]
                for j in range(i, i+2):
                    H[j, i] += 2 * fi
                    H[j, i+1] += 2 * fi
                    H[-1, j] += 2 * fi
                    H[j, -1] += 2 * fi
                H[-1, -1] += 2 * fi
                
        H = H.tocsc()
        for i in range(n):
            H += A_col[:, i].dot(A_col[:, i].transpose())
            
        return 2 * H
    
    else:
        raise ValueError("Wrong function inputs.")
    










    
    
    
    
    
    
    
        