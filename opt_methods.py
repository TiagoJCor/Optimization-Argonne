"""
Implementation of different optimization algorithms. 
"""

import time
import numpy as np

import nlopt

from scipy.linalg import block_diag
from scipy.optimize import fmin_l_bfgs_b


def grad_desc(f, g, x0, alpha=1e-4, max_step=1e4, tol=1e-6, verbose=False):
    """Sequential gradient descent algorithm with fixed step size alpha. 
    
    Args:
        f (func): Mathematical function to be minimized.
        g (func): Gradient of f.
        x0 (ndarray): Initial guess.
        alpha (float): Learning rate.
        max_step (int): Maximum number of iterations.
        tol (float): Value of the gradient norm at which the algorithm stops.

    Returns:
        x_min (ndarray): Minimizer found.
        f(x_min) (float): Minimum found.
    """
    
    if (verbose == True):
        print("\nGradient descent initialized at \nx = %s \nf(x) = %s\n"
              "||g(x)|| = %s\n" % (x0, f(x0), np.linalg.norm(g(x0))))
    
    x_min = x0.copy()
    k = 0
    x_list = [x_min.copy()]
    
    for s in range(int(max_step)):
        x_min = x_min - alpha * g(x_min)
        k += 1
        x_list.append(x_min.copy())
        
        if(np.linalg.norm(g(x_min)) < tol):
            print("Gradient is below threshold, routine stopped.")
            break
    
    if (verbose == True):  
        print("After %d steps we have \nx = %s \nf(x) = %s\n||g(x)|| = %s" % (k, 
               x_min, f(x_min), np.linalg.norm(g(x_min)))) 
    
    x_list = np.asarray(x_list).transpose()
    return x_min, f(x_min), x_list


def block_cd(block_choose, f, g, b, x0, alpha=1e-4, max_step=1e4, tol=1e-6, 
             verbose=False):
    """Block coordinate descent algorithms. Ways of choosing blocks:
        
    _'cyc_bcd' - Cyclic BC from ShiLiu2016, where each block has n/b adjacent 
    variables.
    e.g. n = 10, b = 2, the first block has vars x1 through x5.
    
    _'cards_deal' - Variables are distributed in a "cards dealing" way, i.e.
    each block gets a variable per round of the distrubution procedure, in order.
    e.g. n = 10, b = 2, the first block has vars x1, x3, x5, x7, x9
    
    _'rand_wo_replace' - Variables are distributed randomly across the b blocks, 
    making sure that in each step all coordinates are used.
    If n mod p != 0, the last block will be bigger (by n mod p).
    
    _'rand_replace' - Variables are distributed randomly across the b blocks, 
    making sure that in each step all coordinates are used.
    If n mod p != 0, the last block will be bigger (by n mod p).
    
    Args:
        f (func): Mathematical function to be minimized.
        g (func): Gradient of f.
        b (float): Number of blocks.
        x0 (ndarray): Initial guess.
        alpha (float): Stepsize.
        max_step (int): Maximum number of iterations.
        tol (float): Value of the gradient norm at which the algorithm stops.

    Returns:
        x_min (ndarray): Minimizer found.
        f(x_min) (float): Minimum found.
    """
    
    if (verbose == True):
        print("\nCyclic coordinate descent initialized at \nx = %s \nf(x) = %s\n"
          "||g(x)|| = %s\n" % (x0, f(x0), np.linalg.norm(g(x0))))
    
    x_min = x0.copy() 
    n = np.size(x_min)   
    
    # Number of variables per block (if n mod b != 0, round down to 
    # nearest integer). 
    n_var_b = int(n/b)
    k = 0
    I = np.eye(n)
    
    for s in range(int(max_step)):
        rand_index = np.random.permutation(n)
        for i in range(b):
            
            if (block_choose == 'cyc_bcd'):
                if (i == (b - 1)):
                    ind = np.arange((b - 1) * n_var_b, n)
                else:
                    ind = np.arange(i * n_var_b, (i + 1) * n_var_b)
            elif (block_choose == 'cards_deal'):
                ind = np.arange(n)[i::b]
                
            elif (block_choose == 'rand_wo_replace'):
                # Last block bigger.
                if (i == (b - 1)):
                    ind = rand_index[(b - 1) * n_var_b:]
                else:
                    ind = rand_index[i * n_var_b:(i + 1) * n_var_b]
                    
            elif (block_choose == 'rand_replace'):
                if (i == (b - 1)):
                    ind = np.random.randint(n, size=(n_var_b + n % b))
                else:
                    ind = np.random.randint(n, size=n_var_b)
                    
            # Compute gradient of ith block variables
            grad_i = g(x_min)[ind]
            desc_vec = np.matmul(I[:, ind], grad_i.reshape(-1, 1)).flatten()
            x_min = x_min - alpha * desc_vec
        k += 1
#        x_list.append(x_min.copy())
        if(np.linalg.norm(g(x_min)) < tol):
            print("Gradient is below threshold, routine stopped.")
            break
    if (verbose == True):    
        print("After %d steps we have \nx = %s \nf(x) = %s\n||g(x)|| = %s" % (k, 
          x_min, f(x_min), np.linalg.norm(g(x_min)))) 
        
    return f(x_min), np.linalg.norm(g(x_min))

def rand_cd(f, g, x0, alpha=1e-4, max_step=1e4, tol=1e-6, verbose=False):
    """Randomized coordinate descent algorithm with fixed step size. 
    
    Args:
        f (func): Mathematical function to be minimized.
        g (func): Gradient of f.
        x0 (ndarray): Initial guess.
        alpha (float): Learning rate.
        max_step (int): Maximum number of iterations.
        tol (float): Value of the gradient norm at which the algorithm stops.

    Returns:
        x_min (ndarray): Minimizer found.
        f(x_min) (float): Minimum found.
        x_list (ndarray): List of points evaluated (as columns).
    """
    
    if (verbose == True):
        print("\nSequential coordinate descent initialized at \nx = %s \nf(x) = %s\n"
          "||g(x)|| = %s\n" % (x0, f(x0), np.linalg.norm(g(x0))))
        
    x_min = x0.copy()
    n = np.size(x_min)
    k = 0
    x_list = [x_min.copy()]
    
    for s in range(int(max_step)):
        # Randomly pick a single coordinate 1 <= i <= n.
        i = np.random.randint(n)
        # Move along the ith coordinate.
        x_min[i] -= alpha * g(x_min)[i]
        k += 1
        x_list.append(x_min.copy())
        
        if(np.linalg.norm(g(x_min)) < tol):
            print("Gradient is below threshold, routine stopped.")
            break
        
    if (verbose == True):
        print("After %d steps we have \nx = %s \nf(x) = %s\n||g(x)|| = %s" % (k, 
               x_min, f(x_min), np.linalg.norm(g(x_min)))) 
    
    # Convert to numpy array with vectors as columns.
    x_list = np.asarray(x_list).transpose()
    return x_min, f(x_min), x_list


def pvd_subprob(f, g, D_k_l, x, n, p, l, max_sub_eval, eps1, verbose, turn=1):
    """Auxiliary function for inexact_pvt.
    Computes inexact solution to the optimization subproblem.
    
    Args:
        f (func): Mathematical function to be minimized.
        g (func): Gradient of f.
        D_k_l (ndarray): Matrix representing allowed directions of change for 
                         coordinates not assigned to this processor.
        x (ndarray): Starting point for subproblem optimization routine.
        n (int): Number of dimensions.
        p (int): Total number of processors.
        l (int): Index of this particular processor.
        max_sub_eval (int): Maximum number of function evaluations.
        eps1 (float): Tolerance.
        verbose (int): Determines number of print statements (0, 1 or 2).

    Returns:
        x_l_min (ndarray): Minimizer found.
        num_evals (int): Number of function evaluations done.
        time_per_f_eval (float): Time elapsed per function evaluation.
        time_per_linalg (float): Time elapsed per step of the optimization
                                 subroutine.
    """
    
    if(verbose == 2):
        print("\nInside subproblem: routine started, "
              "function value is:\n", f(x), "\nwith gradient:\n", g(x), "\n")
        
    # Number of variables in this processor.
    n_var_p = int(n/p)
    # Indices of variables assigned to this processor.
    ind_l = np.arange(l * n_var_p, (l + 1) * n_var_p)    
    # Indices of variables that are only allowed to change along directions
    # specified by the columns of D_k_l.
    ind_lbar = np.delete(np.arange(n), ind_l)
    # Segment the x vector 
    x_l = x[ind_l]
    x_lbar = x[ind_lbar]
    
    # Starting vector for optimization routine.
    w0 = np.concatenate((x_l, np.zeros(p-1)))
    # Calculate initial gradient - need this for stopping condition (sub_tol)
    x_init = np.zeros(n)
    grad_init = np.zeros(n_var_p + p - 1)
    x_init[ind_l] = w0[:n_var_p]
    x_init[ind_lbar] = x_lbar + np.matmul(D_k_l, w0[n_var_p:])
    g_init = g(x_init)
    grad_init[:n_var_p] = g_init[ind_l]
    grad_init[n_var_p:] = np.matmul(g_init[ind_lbar].reshape(1, -1), D_k_l)
    grad_init_norm = np.linalg.norm(grad_init)
    
    sub_tol = turn * max(grad_init_norm/15, 0.5*eps1)
    
#### Alternative to using nlopt library, using scipy's 
#### fmin_l_bfgs_b function. 
#    
#    
#    def func(w):
#        x_eval = np.zeros(n)
#        # x_l 
#        x_eval[ind_l] = w[:n_var_p]
#        # x_lbar + D*lambda_lbar
#        x_eval[ind_lbar] = x_lbar + np.matmul(D_k_l, w[n_var_p:])
#        if (verbose == 1):
#            print("Inside func: func val =", f(x_eval)) 
#        return f(x_eval)
#
#    def grad_func(w):
#        x_eval = np.zeros(n)
#        # x_l 
#        x_eval[ind_l] = w[:n_var_p]
#        # x_lbar + D*lambda_lbar
#        x_eval[ind_lbar] = x_lbar + np.matmul(D_k_l, w[n_var_p:])
#        grad = np.zeros(n_var_p + p - 1)
#        f_grad = g(x_eval)
#        grad[:n_var_p] = f_grad[ind_l]
#        grad[n_var_p:] = np.matmul(f_grad[ind_lbar].reshape(1, -1), D_k_l)
#        if (verbose == 1):
#            print("Inside grad: grad norm =", np.linalg.norm(grad))
#        return grad
#        
#    w0 = np.concatenate((x_l, np.zeros(p-1)))
#    
#    # Can also use minimize(method=’L-BFGS-B’) - same algorithm.
#    # See scipy documentation for differences in function parameters.
#    (w_min, fw_min, d) = fmin_l_bfgs_b(func, w0, fprime=grad_func, 
#                                              pgtol=sub_tol, maxfun=max_sub_eval)
#    num_evals = d['funcalls']
#    if (verbose > 0):
#        print("Return flag:", d['warnflag'])
#        print("Number of evaluations of f in subproblem:", num_evals)
#    x_l_min = np.zeros(n)
#    # x_l 
#    x_l_min[ind_l] = w_min[:n_var_p]
#    # x_lbar + D*lambda_lbar
#    x_l_min[ind_lbar] = x_lbar + np.matmul(D_k_l, w_min[n_var_p:])
#    return x_l_min, num_evals

    f_eval_time = [0]
    # list will store start (i=0) and interval (i=2,3,...) times
    lin_alg_time = [0]

    if (verbose > 0):
        print("grad_init_norm/15:", grad_init_norm/15, "eps1:", eps1, "sub_tol:",
              sub_tol)
        print("    we are optimizing", np.size(w0), "variables here!")

    first_step = [True]
    store_wmin = np.zeros(n_var_p + p - 1)
    
    def phi(w, grad):
        """Evaluates f for subproblem optimization. 
        Variables that are being optmized: x_l and lambda_lbar.
        w: concatenation of x_l and lambda_lbar, of size n_var_p + p - 1.
        grad: (n_var_p + p - 1,) ndarray
        """
        
        # check whether first start time has been logged
        if (lin_alg_time[0] != 0):
            lin_alg_time.append(time.time() - lin_alg_time[0])
        
        t0 = time.time()
        x_eval = np.zeros(n)
        # x_l 
        x_eval[ind_l] = w[:n_var_p]
        # x_lbar + D*lambda_lbar
        x_eval[ind_lbar] = x_lbar + np.matmul(D_k_l, w[n_var_p:])
        
        if (grad.size > 0):
            stop_cond = (np.linalg.norm(grad) < sub_tol) \
                         and (first_step[-1] == False) \
                         or (len(first_step)) == max_sub_eval
            if stop_cond:
                grad[:] = 0
                # Store both the minimizer and the minimum.
                store_wmin[:] = w
                first_step.append(f(x_eval))
                # This stops the optimization routine.
                return -np.inf

            # (n,) array - full gradient of f at x_eval
            f_grad = g(x_eval)
            t1 = time.time()
            grad[:n_var_p] = f_grad[ind_l]
            grad[n_var_p:] = np.matmul(f_grad[ind_lbar].reshape(1, -1), D_k_l)
            first_step.append(False)
        
        f_x_eval = f(x_eval)
        t1 = time.time()
        f_eval_time[0] += t1-t0
        
        if (verbose > 0):
            print("phi val:", f_x_eval, "gradphi norm:", np.linalg.norm(grad))
            
        lin_alg_time[0] = time.time()
        return f_x_eval
    
    # Choose optimization algorithm. See options at
    # https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
    algo = nlopt.LD_LBFGS
    #algo = nlopt.LD_SLSQP
    opt = nlopt.opt(algo, n_var_p + p - 1)
    # Function to be minimized.
    opt.set_min_objective(phi)
    # Maximum number of steps (taken as input to original function).
    opt.set_ftol_abs(10**-10)
    opt.set_ftol_rel(10**-10)
    opt.set_maxeval(max_sub_eval)
    w_min = opt.optimize(w0)
    minf = opt.last_optimum_value()
    
    # Number of function evaluations performed
    result_value = opt.last_optimize_result()
    num_evals = len(first_step) - 1
    
    # Time-related computations
    if (num_evals > 1):
        time_per_f_eval = f_eval_time[0] / (num_evals - 1)
        time_per_linalg = sum(lin_alg_time[1:]) / (num_evals - 1)
    else:
        time_per_f_eval = -1
        time_per_linalg = -1
    
    # Test whether the gradient norm condition got enforced.
    # If so, correct the values of the minimum and minimizer.
    if (isinstance(first_step[-1], bool) == False):
        w_min = store_wmin[:]
        minf = first_step[-1]
    
    if (verbose > 0):
        print("result value:", result_value)
        print("num_evals:", num_evals, "max_sub_eval:", max_sub_eval)
    # For tol-based tests, to warn you regardless of verbose setting
    if(num_evals == max_sub_eval - 1):
         print("tolerance NOT reached, result code", result_value)
         
    if(verbose == 2):
        print("Resuls:\n")
        if (not first_step[-1]):
            print("optimum:", w_min)
            print("minimum value:", minf)
            # note - if sub_grad is too low, the result code might be 1
            print("tolerance NOT reached (result value 5)")
            print("actual result value:", result_value, "\n")
        else:
            print("optimum:", store_wmin)
            print("minimum value:", first_step[-1])
            print("tolerance WAS reached (result value 2)")
            print("actual result value:", result_value, "\n")
    # Note on result codes: https://nlopt.readthedocs.io/en/latest/ 
    # /NLopt_Reference/#successful-termination-positive-return-values
        
    # Recreate the value of x given the optimization result.
    x_l_min = np.zeros(n)
    # x_l 
    x_l_min[ind_l] = w_min[:n_var_p]
    # x_lbar + D*lambda_lbar
    x_l_min[ind_lbar] = x_lbar + np.matmul(D_k_l, w_min[n_var_p:])
    
    return x_l_min, num_evals, time_per_f_eval, time_per_linalg

def inexact_pvt(f, g, p, x0, max_step=1e3, max_sub_eval=100, eps1=1e-3, 
                eps2=1e-6, verbose=0, method="pvd"):
    """Parallel variable transformation algorithm with inexact subproblem
    solution from Solodov1997. Divides the n variables equally across p
    procesors. Requires n mod p = 0.
    
    The optimization subroutines (in each processor) are performed using
    the l-bfgs algorithm from nlopt.
    
    Args:
        f (func): Mathematical function to be minimized.
        g (func): Gradient of f.
        p (float): Number of processors.
        x0 (ndarray): Initial guess.
        max_step (int): Maximum number of outer iterations.
        max_sub_eval (int): Maximum number of subproblem function evaluations.
        eps1 (float): Tolerance for gradient norm of main problem opt.
        eps2 (float): Tolerance for change in function values in subsequent
                      outer iterations.
        verbose (int): Controls amount of print statements (0, 1 or 2).
        method (str): "pvd" or "jacobi", depending on how you want the
                      "forget me not" term of not. Extra option: "pvd2", where
                      subproblem is called twice for each outer iteration.
                      In the first one we determine the maximum number of 
                      function evals, in the second force all processors to
                      run until that number of function evals is reached.

    Returns:
        k (int): Number of 
        max_evals_list (list): 
        f_min_list (list): Function minimum at the end of each iteration.
        g_norm_list (list): Gradient norm at the end of each iteration.
        avg_t_f_eval (float): Average of function evaluation times.
        avg_t_linalg (float): Average of optimization subroutine linear
                              algebra operations times.
    """
    
    if (verbose > 0):
        print("\nInexact PVD initialized at \nx = %s \nf(x) = %s\n"
          "||g(x)|| = %s\n" % (x0, f(x0), np.linalg.norm(g(x0))))
        
    x_min = x0.copy() 
    n = np.size(x_min)   
    
    if (n % p != 0):
        raise ValueError("Need n to be divisible by p.")
        
    if (type(p) != int):
        raise ValueError("The value of p must be a positive integer.")
        
    # Number of variables per block.
    n_var_p = int(n/p)
    outer_iter = 0
    tol_break = False
    # store max number of subproblem steps per PVT step
    max_evals_list = []
    # stores minimum found at end of each iteration, starting at f(x0)
    f_min_list = [f(x0),]
    # same as above, for grad norm
    g_norm_list = [np.linalg.norm(g(x0)),]
    # stores avg time of evaluating f per iteration
    avg_t_f_eval = []
    # stores avg time of subproblem linalg operations per iteration
    avg_t_linalg = []
    
    print("\np =", p)
    
    for s in range(int(max_step)):
        if (verbose > 0):
            print("\n INITIATING MASTER STEP", s+1, "\n")
            
        # Directions for "forget-me-not" term.
        if ("pvd" in method):
            d_k = -g(x_min) / np.linalg.norm(g(x_min))
            
            if (method == "pvd2"):
                passes = 2
            else:
                passes = 1
            
        elif (method == "jacobi"):
            d_k = np.zeros(n)
            
        else:
            raise ValueError("Choose a valid method.")
        # Break d_k into p blocks of n/p variables & put them in the diagonal
        # of D_k, a n-by-p matrix.
        x_k_list = []
        
        t_f_eval = np.zeros(p)
        t_linalg = np.zeros(p)
        num_evals_s = np.zeros(p)
        # In each processor...PART 1
        for l in range(p):
            
            # First delete lth coordinate from d_k\
            d_k_l = np.delete(d_k, np.arange(l * n_var_p, (l + 1) * n_var_p))
            # Break d_k into p blocks of n/p variables & put them in the diag
            # of D_k, rendering a n_\bar{l}-by-(p-1) matrix.
            if (p == 1):
                D_k_l = np.array([])
            else:
                D_k_l = block_diag(*np.split(d_k_l.reshape(-1, 1), p-1))

            # Solve subproblem....
            if (verbose > 0):
                print("Subproblem n", l, "called with: f =", f(x_min), 
                      "||g||=", np.linalg.norm(g(x_min)))
            

            res = pvd_subprob(f, g, D_k_l, x_min, n, p, l, max_sub_eval, 
                              eps1, verbose)
            (x_l_min, num_evals_s[l], t_f_eval[l], t_linalg[l]) = res
                
            
            if (verbose > 0):
                print("Subproblem n", l, "returned with: f =", f(x_l_min), 
                      "||g||=", np.linalg.norm(g(x_l_min)))
            x_k_list.append(x_l_min)
        
        if (passes == 2):
            x_k_list = []
            t_f_eval = np.zeros(p)
            t_linalg = np.zeros(p)
            num_evals_max = int(max(num_evals_s))
            for l in range(p):
            
                # First delete lth coordinate from d_k\
                d_k_l = np.delete(d_k, np.arange(l * n_var_p, (l + 1) * n_var_p))
                # Break d_k into p blocks of n/p variables & put them in the diag
                # of D_k, rendering a n_\bar{l}-by-(p-1) matrix.
                if (p == 1):
                    D_k_l = np.array([])
                else:
                    D_k_l = block_diag(*np.split(d_k_l.reshape(-1, 1), p-1))
    
                # Solve subproblem....
                if (verbose > 0):
                    print("Subproblem n", l, "called with: f =", f(x_min), 
                          "||g||=", np.linalg.norm(g(x_min)))
                
    
                res = pvd_subprob(f, g, D_k_l, x_min, n, p, l, num_evals_max, 
                                  0, verbose, turn=0)
                (x_l_min, num_evals_s[l], t_f_eval[l], t_linalg[l]) = res
                    
                
                if (verbose > 0):
                    print("Subproblem n", l, "returned with: f =", f(x_l_min), 
                          "||g||=", np.linalg.norm(g(x_l_min)))
                x_k_list.append(x_l_min)
                
                
        # Make sure a time measurement was actually made (i.e. num_evals > 1)
        if t_f_eval[0] != -1:  
            avg_t_f_eval.append(np.average(t_f_eval, weights=num_evals_s))
            avg_t_linalg.append(np.average(t_linalg, weights=num_evals_s))
        
        # Convert to numpy array with vectors as columns.
        x_k_list = np.asarray(x_k_list).transpose()
        f_list = np.apply_along_axis(f, 0, x_k_list)
        
        # Store max number of func evals done by processors
        max_evals_list.append(np.max(num_evals_s))
        
        # Candidate for new min
        f_min_new = f(x_k_list[:, np.argmin(f_list)])
        # Current min
        f_min_now = f(x_min)
        
        # Synchronization step...choose value of x that gives best min
        if (f_min_now > f_min_new):
            x_min = x_k_list[:, np.argmin(f_list)]
            rel_change = (f_min_now - f_min_new) / (1 + np.abs(f_min_now))
            f_min_list.append(f_min_new)
            if (verbose > 0):
                print("\nMin value at sync: ", f_min_new)
        else:
            rel_change = 0
            f_min_list.append(f_min_now)
            if (verbose > 0):
                print("\nBetter min not found, value not updated")
                
        outer_iter += 1
        # Test stopping conditions
        cond1 = (np.linalg.norm(g(x_min)) < eps1)
        cond2 = (rel_change < eps2)
        
        g_norm_list.append(np.linalg.norm(g(x_min)))
        
        if(cond1 and cond2):
            tol_break = True
            print("PVT stopping conditions satisfied! Routine stopped after", 
                  s+1, "iterations.")
            break
        
    if (verbose > 1):    
        print("After %d steps we have \nf(x) = %s\n||g(x)|| = %s" % 
              (outer_iter, f(x_min), np.linalg.norm(g(x_min)))) 
    if (tol_break == False):
        print("max_step of", max_step, "did not bring grad(f) to below tol.")
    
    print("fmin =", f_min_list[-1], "gradnorm =", np.linalg.norm(g(x_min)))
    
    ret = (outer_iter, max_evals_list, f_min_list, g_norm_list, 
           sum(avg_t_f_eval)/len(avg_t_f_eval), 
           sum(avg_t_linalg)/len(avg_t_linalg))
    
    return ret