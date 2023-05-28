# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:17:58 2023

@author: Ivan
"""

import numpy as np
from ivim_b_value_crlb_cost_function import cost_function
import matplotlib.pyplot as plt
from tqdm import tqdm

def cost_evaluation(params, b_values=[0, 100, 200, 800], b_values_list=np.linspace(0,1000,101)):
    costs = []
    f_terms = []
    Dp_terms = []
    Dt_terms = []
    sum_factors = []
    for b_value in b_values_list:
        b_values_eval = b_values + [b_value]
        
        # Evaluate cost
        cost, sum_factor, f_term, Dp_term, Dt_term = cost_function(b_values_eval, params, return_cost_terms=True)
        
        # Append results to lists
        costs.append(cost)
        f_terms.append(f_term)
        Dp_terms.append(Dp_term)
        Dt_terms.append(Dt_term)
        sum_factors.append(sum_factor)
    return costs, f_terms, Dp_terms, Dt_terms

def cost_map(f, Dp, Dt, N_param, var_param, b_values=[0, 100, 200, 800]):    
    f_map = np.zeros((len(b_values_list),N_param))
    Dp_map = np.zeros((len(b_values_list),N_param))
    Dt_map = np.zeros((len(b_values_list),N_param))
    total_cost_map = np.zeros((len(b_values_list),N_param))
    
    for i in range(N_param):
        if var_param == "f":
            costs, f_terms, Dp_terms, Dt_terms = cost_evaluation(params=[f[i], Dp, Dt], b_values=b_values)
        elif var_param == "D*":
            costs, f_terms, Dp_terms, Dt_terms = cost_evaluation(params=[f, Dp[i], Dt], b_values=b_values)
        elif var_param == "D":
            costs, f_terms, Dp_terms, Dt_terms = cost_evaluation(params=[f, Dp, Dt[i]], b_values=b_values)
        
        f_map[:,i] = f_terms
        Dp_map[:,i] = Dp_terms
        Dt_map[:,i] = Dt_terms
        total_cost_map[:,i] = costs#np.sqrt(np.array(f_terms) + np.array(Dp_terms) + np.array(Dt_terms))
        
    if var_param == "f":
        var_lst = f
    elif var_param == "D*":
        var_lst = Dp
    elif var_param == "D":
        var_lst = Dt
    
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0,0].pcolormesh(var_lst, b_values_list, f_map)
    axs[0,0].set_title("f CV")
    axs[0,0].set_xlabel(var_param)
    axs[0,0].set_ylabel("b-value")
    
    axs[0,1].pcolormesh(var_lst, b_values_list, Dp_map)
    axs[0,1].set_title("D* CV")
    axs[0,1].set_xlabel(var_param)
    axs[0,1].set_ylabel("b-value")
    
    axs[1,0].pcolormesh(var_lst, b_values_list, Dt_map)
    axs[1,0].set_title("D CV")
    axs[1,0].set_xlabel(var_param)
    axs[1,0].set_ylabel("b-value")
    
    axs[1,1].pcolormesh(var_lst, b_values_list, total_cost_map)
    axs[1,1].set_title("Total cost")
    axs[1,1].set_xlabel(var_param)
    axs[1,1].set_ylabel("b-value")
    fig.tight_layout()

params = [0.3, 0.05, 0.001]
b_values_list = np.linspace(0,1000,101)
b_values = [0, 100, 200, 300, 800] # Standardvärden

costs, f_terms, Dp_terms, Dt_terms = cost_evaluation(params)

# Plot the results
"""
fig, axs = plt.subplots(ncols=4)
axs[0].plot(b_values_list, costs, ls="-", marker=".", color="black", label="Total cost")
axs[0].set_title("Total cost")

axs[1].plot(b_values_list, f_terms, ls="-", marker=".", color="blue", label="f term")
axs[1].set_title("f CV")

axs[2].plot(b_values_list, Dp_terms, ls="-", marker=".", color="red", label="Dp term")
axs[2].set_title("D* CV")

axs[3].plot(b_values_list, Dt_terms, ls="-", marker=".", color="green", label="Dt term")
axs[3].set_title("D CV")
fig.tight_layout()
"""

N_param = 101

f_var = np.linspace(0.1,0.2, N_param)
Dp_var = np.linspace(0.03, 0.07, N_param)
Dt_var = np.linspace(0.0004, 0.001, N_param)
f_fix = 0.15
Dp_fix = 0.05
Dt_fix = 0.0007

b_values = [0, 20, 120, 120, 120, 120, 800, 800, 800, 800]
cost_map(f_var, Dp_fix, Dt_fix, N_param, var_param="f", b_values=b_values)
cost_map(f_fix, Dp_var, Dt_fix, N_param, var_param="D*", b_values=b_values)
cost_map(f_fix, Dp_fix, Dt_var, N_param, var_param="D", b_values=b_values)