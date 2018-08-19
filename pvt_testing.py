"""
Testing of the Parallel Variable Transformation algorithms - PVD, Jacobi.

Choose values for the PVT parameters, run simulations across all 
combinations of values, and save results to CSV file.
"""

import csv
import test_functions as tfu
import opt_methods as opt

from itertools import product

################################################

# tfu.rosen_ext, tfu.powell_ext, tfu.trig_func, tfu.arrowhead, tfu.nondiag_quartic, 
# tfu.disc_int_func, tfu.shanno_quad, tfu.broyden_tri

func_list = [tfu.shanno_quad]
n_list = [1024]
p_list = [1,2,4,8,16,32]
eps1_list = [1e-4]
eps2_list = [1e-5]
max_step = 200
max_sub_eval = 300
# jacobi or pvd or pvd2
method = "pvd"
save = True


################################################

func_names = [func.__name__ for func in func_list]
func_d_list = [getattr(tfu, name + "_d") for name in func_names]

# loop over all variable lists
for (f, g), n, p, eps1, eps2 in list(product(zip(func_list, func_d_list), 
                                     n_list, p_list, eps1_list, eps2_list)):
    
    try:
        res = opt.inexact_pvt(f, g, p, tfu.starting_x(f, n), max_step, 
                              max_sub_eval, eps1, eps2, verbose=0, 
                              method=method)
        # unpack results 
        (n_outer, max_evals_list, f_min_list, g_norm_list) = res[:4]
        (avg_t_feval, avg_t_linalg) = res[4:]
        
        print("\nIn this iteration of the loop:\n", f.__name__, g.__name__, 
                                                    n, p, eps1, eps2)
        
        print("\nTime stuff:", avg_t_feval, avg_t_linalg)
        
        if save:
            
            file_name = method + "_tests.csv"
                
            with open(file_name, 'a', newline='') as csvfile:
                
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', 
                                        quoting=csv.QUOTE_MINIMAL)
                
                inner_log = [i for el in zip(max_evals_list, f_min_list[1:], 
                                             g_norm_list[1:]) 
                               for i in el]
                if n_outer < max_step:
                    row_filler = [-1] * 3 * (max_step - n_outer)
                else:
                    row_filler = []
                row_to_write = [f.__name__, n, p, eps1, eps2, avg_t_feval, 
                                avg_t_linalg, int(n_outer), f_min_list[0], 
                                g_norm_list[0]] + inner_log + row_filler
                
                spamwriter.writerow(row_to_write)
    except Exception as e:
        print(e)



    
    
    
    
    
    
    
    
    
    
    
    
    