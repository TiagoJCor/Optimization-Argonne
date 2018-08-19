"""
Finite difference tests of gradients and Hessians.
"""

import numpy as np

from scipy.sparse import issparse

def test_Hess(g, n, h=1e-8, tol=1e-5):
    """Computes finite difference approximation to the Hessian of f at a few
    distinct points (including x0) and compares it to the Hessian calculated
    by g.
    
    Args:
        h (float): Spacing in finite difference computations.
        tol (float): Relative tolerance when comparing gradients.
        
    Returns: 
        success (bool): True if the gradients match for all points tested.
    """

    success = 1
    
    for k in range(1,4):
        x = 5 * k * (np.random.rand(n) - 0.5) 
        
        hess = g(x, out='Hess')
        
        if issparse(hess):
            hess = hess.todense()
            
        hess_approx = np.zeros((n, n))
        I = np.eye(n)
        for j in range(n):
            for i in range(j+1):
                ej = I[:, j]
                # Finite difference approximatin to the Hessian.
                hess_approx[i, j] = (g(x + h*ej)[i] - g(x)[i]) / h
                hess_approx[j, i] = hess_approx[i, j]
                
        success *= np.allclose(hess, hess_approx, atol=tol)
        print(hess, "\n", hess_approx, "\n\n")
    return success

def test_grad(f, g, n, h=1e-8, tol=1e-5):
    """Computes finite difference approximation to the gradient of f at a few
    distinct points (including x0) and compares it to the gradient calculated
    by g.
    
    Args:
        h (float): Spacing in finite difference computations.
        tol (float): Relative tolerance when comparing gradients.
    Returns: 
        success (bool): True if the gradients match for all points tested.
    """
    print(h, tol)
    success = 1
    
    for i in range(1,4):
        x = 5 * i * (np.random.rand(n) - 0.5) 
        grad = g(x)
        grad_approx = np.zeros(n)
        I = np.eye(n)
        for j in range(n):
            ej = I[:, j]
            grad_approx[j] = (f(x + h*ej) - f(x)) / h
            
        success *= np.allclose(grad, grad_approx, atol=tol)
        
        print(grad, grad_approx, "\n")
        
    return success









