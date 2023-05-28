# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:39:54 2023

@author: Ivan
"""

import numpy as np
import scipy.optimize
import ivim_b_value_crlb_cost_function

params = [1000, 0.1, 0.02, 5e-4]
params = [1000, 0.1, 0.02, 5e-4]
NEX = 14
initial_guess = np.linspace(0,800,NEX)
bounds = [(0,800) for i in range(len(initial_guess))]
res = scipy.optimize.minimize(ivim_b_value_crlb_cost_function.cost_function_S0_no_covar, \
                              initial_guess, args=(params), bounds=bounds, method="Nelder-mead")

# Get the optimal b-values rounded to nearest 10
opt_b_values = [round(int(b_value),-1) for b_value in res.x]

# Construct an array with the unique b_values, and an array of their corresponding NEX
unique_b_values, nex_list = np.unique(opt_b_values, return_counts=True)

# Print the results
for i in range(len(unique_b_values)):
    print(unique_b_values[i], nex_list[i])