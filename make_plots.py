"""Makes plots of function values, gradient norms and time data stored in CSV 
file."""

import csv
import os.path
import matplotlib.pyplot as plt
import numpy as np

from itertools import cycle

################################################

file_name = 'pvd_tests.csv'

# 'rosen_ext', 'powell_ext', 'trig_func', 'arrowhead', 
# 'nondiag_quartic', 'shanno_quad', 'disc_int_func'

# choose values to plot
plot_f_name = ['shanno_quad']
plot_n = [4096]
plot_p = [1,2,4,8,16,32]
# want to plot function values ("fmin") or gradient norms ("gnorm")?
what2plot = "gnorm"
# or want to plot time? ("True" overrides the above)
plot_time = True
# if the above is True, do you wish to plot the function evaluation ("feval") 
# or linear algebra ("linalg") times?
time_opt = "linalg"
# want to save the plot?
save = True
# want to display the plot on the console?
display = True
# image dpi
img_dpi = 200

################################################


# different plot for each function
for f_to_plot in plot_f_name:
    for n_to_plot in plot_n:
        
        plotted_p = []
        
        colors = cycle(['g', 'b', 'r', 'k', 'm', 'c'])
        markers = cycle(['*', 'o', '.', 'v', '1'])
        linestyle = cycle([':', '-.', '--', '-'])
        
        with open(file_name, newline='') as csvfile:
            # skip header
            next(csvfile)
            
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            
            plt.figure(dpi=img_dpi)
            ax = plt.subplot(111)
            
            if not plot_time:
            
                for row in spamreader:         
                    (f_name, n, p, eps1, eps2, avg_t_feval,
                     avg_t_linalg, n_outer) = row[:8]
                    
                    p = int(p)
                    
                    # only plot the specified parameters ignore repeated p's
                    if (f_name == f_to_plot) and (int(n) == n_to_plot) and \
                       (p in plot_p) and (p not in plotted_p): 
                           
                        f_min_list = [float(val) for i, val in enumerate(
                                      row[8:]) if i % 3 == 0 and \
                                      float(val) > -1] 
                            
                        g_norm_list = [float(val) for i, val in enumerate(
                                      row[8:]) if i % 3 == 1 and \
                                      float(val) > -1] 
                        
                        max_eval_list = [0] + [int(float(val)) for i, val in 
                                        enumerate(row[8:]) if i % 3 == 2 and 
                                        float(val) > -1]
                        
                        plotted_p.append(p)
                        col = next(colors)
                        mark = next(markers)
                        lin = next(linestyle)
                        
                        if (what2plot == "fmin"):
                            ydata = f_min_list
                            ylabel = "minimum of f"
                            
                        elif (what2plot == "gnorm"):
                            ydata = g_norm_list
                            ylabel = "gradient norm"
                            
                        else:
                            raise ValueError("Invalid plotting options.")
                        
                        plt.semilogy(np.cumsum(np.array(max_eval_list)), 
                                     ydata, lin + mark, c=col, 
                                     label=f"p = {p}")
                        
                if len(plotted_p) > 0:
                    plt.xlabel("func evals")
                    plt.ylabel(ylabel)
                    plt.title(f_to_plot + f", n = {n_to_plot}")
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
                    if save:
                        
                        plot_id = f_to_plot + "_n_" + f"{n_to_plot}"
                        
                        if (file_name == "pvd_tests.csv"):
                            folder = "pvd_plots/"
                            
                        elif (file_name == "jacobi_tests.csv"):
                            folder = "jacobi_plots/"
                            
                        elif (file_name == "pvd2_tests.csv"):
                            folder = "pvd2_plots/"
                            
                        else:
                            pass
                        
                        plot_file_name = folder + what2plot + "/" + \
                                         what2plot + "_" + plot_id + ".png"
                                  
                        # check whether file exists before saving
                        if not os.path.isfile(plot_file_name):
                            plt.savefig(plot_file_name, bbox_inches="tight")
    
                    if display:
                        plt.show()
                        
                    else:
                        plt.close()
                    
                else:
                    plt.close()
                            
            else:
                
                p_plot_list = []
                y_plot_list = []
                
                for row in spamreader:         
                    (f_name, n, p, eps1, eps2, avg_t_feval, 
                     avg_t_linalg, n_outer) = row[:8]
                    
                    p = int(p)
                    avg_t_feval, avg_t_linalg = map(float, (avg_t_feval, 
                                                            avg_t_linalg))  
                    
                    # only plot the specified parameters ignore repeated p's
                    if (f_name == f_to_plot) and (int(n) == n_to_plot) and \
                       (p in plot_p): 
                        
                        if (time_opt == "feval"):
                            y_data = avg_t_feval
                        
                        elif (time_opt == "linalg"):
                            y_data = avg_t_linalg
                        
                        else:
                            raise ValueError("Invalid plotting options.")
                        
                        if (y_data > 0):
                            p_plot_list.append(p)
                            y_plot_list.append(y_data)
                        
                # check whether there are any values to plot
                if len(p_plot_list) > 0: 
                    
                    yplot = []
                    
                    # average over p
                    for pi in plot_p:
                        yplot.append(np.mean([i for i, j 
                                              in zip(y_plot_list, p_plot_list) 
                                              if j == pi]))
                        
                    print(plot_p, yplot)
                    plt.semilogy(plot_p, yplot, "ro-")
                    plt.xlabel("p")
                    plt.ylabel("time (sec)")
                    plt.title(f_to_plot + f", n = {n_to_plot}")
            
                    if save:
                        plot_id = f_to_plot + "_n_" + f"{n_to_plot}"
                        
                        if (file_name == "pvd_tests.csv"):
                            folder = "pvd_plots/"
                            
                        elif (file_name == "jacobi_tests.csv"):
                            folder = "jacobi_plots/"
                            
                        elif (file_name == "pvd2_tests.csv"):
                            folder = "pvd2_plots/"
                            
                        else:
                            pass
                        
                        plot_file_name = folder + "time" + "/" + "time" + \
                                         time_opt + "_" + plot_id + ".png"
                        
                        # check whether file exists before saving
                        if not os.path.isfile(plot_file_name):
                            plt.savefig(plot_file_name, bbox_inches="tight")
                    
                    if display:
                        plt.show()
                    else:
                        plt.close()
                else:
                    plt.close()
