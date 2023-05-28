# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:52:26 2023

@author: Ivan
"""

import numpy as np
import ivim_b_value_crlb_cost_function
import scipy.optimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import tomoview
import matplotlib.colors as colors 
import os

plt.style.use("default")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['text.usetex'] = True # Prevents font change when italicized

n = 10
f_values = np.linspace(0.05, 0.5, n)
D_stars = np.linspace(0.005, 0.03, n)
b_idx = list(range(0, n))
D = 1e-3

def print_b_set_nicely(b_set):
    """Prints b-sets nicely where unique b-values are listed next to their number of NEX's.

    Args:
        b_set (array-like): Array of b-sets
    """
    # Construct an array with the unique b_values, and an array of their corresponding NEX
    unique_b_values, nex_list = np.unique(b_set, return_counts=True)

    # Print the results
    print()
    for i in range(len(unique_b_values)):
        print(unique_b_values[i], nex_list[i])
    print()
    
def plot_D_star_volume(volume, f_index=0, f_values=f_values, D_stars=D_stars, b_idx=b_idx):
    # Plotta kartan
    fig, ax = plt.subplots()
    ax.pcolormesh(D_stars*1000, b_idx, volume[:,:,f_index])
    ax.set_xlabel("$D^*$", fontsize=12)
    ax.set_ylabel("b set index", fontsize=12)
    ax.set_title("f = {}".format(f_values[f_index]))
    
def plot_D_star_map(volume, f_index=0, f_values=f_values, D_stars=D_stars, b_idx=b_idx):
    # Plotta kartan
    fig, ax = plt.subplots(figsize=(4,3))
    cbar = ax.pcolormesh(D_stars*1000, b_idx, volume, cmap="gray", norm=colors.LogNorm(vmin=volume.min(), vmax=volume.max()))
    ax.set_xlabel("$D^*$ [µm$^2$/ms]", fontsize=12)
    ax.set_ylabel("b set index", fontsize=12)
    fig.colorbar(cbar, ax=ax, extend="max", label="Cost [a.u.]")
    ax.tick_params(labelsize=12)
    path_report_figures = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Rapport\appendix\figures"
    fig.savefig(os.path.join(path_report_figures, f"Dstar_cost_map_{f_values[f_index]}.pdf"), bbox_inches="tight")

def get_D_star_map(n, D_stars, S0=1000, f=0.1, D=D, return_lists=True):
    
    # Använd CRLB för att ta fram optimala subsets för varje D*
    # Set up optimization parameters
    ivim_params_sets = [[1000, f, D_star, D] for D_star in D_stars]
    NEX = 14
    initial_guess = np.linspace(0,800,NEX)
    bounds = [(0,800) for i in range(len(initial_guess))]
                                                                                                                    # ÄNDRA COVAR HÄR
    b_sets = [scipy.optimize.minimize(ivim_b_value_crlb_cost_function.cost_function_S0, \
                                  initial_guess, args=(ivim_params), bounds=bounds, method="Nelder-mead").x \
              for ivim_params in tqdm(ivim_params_sets, desc="Optimizing for f = {}".format(f))]
    
    # Round the b-values to the nearest 10
    rounded_b_sets = []
    for i in range(len(b_sets)):
        b_set = []
        for j in range(len(b_sets[i])):
            b_set.append(round(int(b_sets[i][j]), -1))
        rounded_b_sets.append(b_set)
        
    # Optimera baklänges
    # För varje b-set, beräkna kostnaden för ivim_parametrarna
    cost_map = np.zeros((n,n))
    for i in range(len(b_sets)):
        for j in range(len(ivim_params_sets)):
            cost_map[i,j] = ivim_b_value_crlb_cost_function.cost_function_S0(rounded_b_sets[i], ivim_params_sets[j])#     ÄNDRA COVAR HÄR
    
    if return_lists:
        #return cost_map, D_stars, rounded_b_sets
        return cost_map, rounded_b_sets
    else:
        return cost_map
     
def get_cost_volume(n=n, return_lists=False):
    """
    Generates a cost volume with D_star on axis 0, b_sets on axis 1, and f on axis 2

    Args:
        n (int, optional): Number of f-values to evaluate for. Defaults to n.
        return_lists (bool, optional): Variable that determines whether the b-sets should be returned or not. Defaults to False.

    Returns:
        _type_: _description_
    """
    volume = np.zeros((n,n,n))
    rounded_b_sets_list = []
    if return_lists:
        for i in range(len(f_values)):
            volume[:,:,i], rounded_b_sets = get_D_star_map(n, D_stars, f=f_values[i], return_lists=True)
            rounded_b_sets_list.append(rounded_b_sets)
        return volume, rounded_b_sets_list
    else:
        for i in range(len(f_values)):
            volume[:,:,i] = get_D_star_map(n, D_stars, f=f_values[i], return_lists=False)
        return volume

def get_optimal_b_set_across_interval(cost_map, rounded_b_sets, f_idx):
    plot_D_star_map(cost_map, f_index=f_idx)
    sum_costs = [np.sum(cost_map[i,:]) for i in range(len(rounded_b_sets))]
    idx = np.where(sum_costs == np.min(sum_costs))[0][0]
    opt_b_set = rounded_b_sets[idx]
    print()
    print("Optimal b-set is at list index {}".format(idx))
    print_b_set_nicely(opt_b_set)
    return rounded_b_sets
    
#rounded_b_sets = get_optimal_b_set_across_interval(*get_D_star_map(n, D_stars, return_lists=True))
#print("--------------")
#for b_set in rounded_b_sets:
#    print_b_set_nicely(b_set)
    
    
def get_optimal_b_set_for_volume(volume, rounded_b_sets_list):
    for f_idx in range(volume.shape[2]):
        print(f"Results for f = {f_values[f_idx]}:")
        get_optimal_b_set_across_interval(volume[:,:,f_idx], rounded_b_sets_list[f_idx], f_idx)
        
volume, rounded_b_sets_list = get_cost_volume(return_lists=True)
get_optimal_b_set_for_volume(volume, rounded_b_sets_list)
#map = get_D_star_map(10, D_stars)