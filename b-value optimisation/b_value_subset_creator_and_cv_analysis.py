# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:35:14 2023

@author: Ivan
"""

import numpy as np
import itertools
from tqdm import tqdm
from ivim_b_value_crlb_cost_function import cost_function
import math

class b_subset:
    def __init__(self, subset):
        self.subset = subset
    
    def calculate_cost(self, ivim_params):
        self.cost = cost_function(subset, ivim_params)
        

b_values_conditioned = [] # List for conditioned subsets
b_values_unconditioned = [] # List for unconditioned subsets
b_values = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 175, 200, 400, 600, 800]

b_values = [0, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 400, 800, 900] # Norge hjärna modifierad
b_values = [0, 15, 30, 45, 60, 100, 150, 200, 300, 800]
b_values *= 1

# Creates lists of all possible combinations with at least 4 b-values
min_no_of_b_values = 7
max_no_of_b_values = 7
params = [0.15, 0.05, 0.001] # Typical prostate IVIM parameters. Results vary with the D* value!
otal=(math.factorial(len(b_values))/(math.factorial(7)*math.factorial(len(b_values)-7)))
for subset_length in range(min_no_of_b_values, max_no_of_b_values+1):
    for subset in tqdm(itertools.combinations(b_values, subset_length)):
        """
        # Only store the 10 subsets with the lowest cost
        if len(b_values_unconditioned) < 10:
            b_values_unconditioned.append(b_sub)
        else:
            costs = [i.cost for i in b_values_unconditioned]
            idx = np.where(costs == max(costs))[0][0]
            if b_sub.cost <= b_values_unconditioned[idx].cost:
                
                b_values_unconditioned.pop(idx)
                b_values_unconditioned.append(b_sub)
        """
        
        # Set conditions for which subsets to include
        if (0 in subset) and \
            (any(b_value >= 400 for b_value in subset)) and \
            (any(b_value <= 100 for b_value in subset if b_value != 0)): #\
            #and (all(b_value <= 900 for b_value in subset)):
                
            b_sub = b_subset(subset)
            b_sub.calculate_cost(params)
            
            # Only store the 10 subsets with the lowest cost
            if len(b_values_conditioned) < 10:
                
                b_values_conditioned.append(b_sub)
            else:
                costs = [i.cost for i in b_values_conditioned]
                idx = np.where(costs == max(costs))[0][0]
                if b_sub.cost < b_values_conditioned[idx].cost:
                    
                    b_values_conditioned.pop(idx)
                    b_values_conditioned.append(b_sub)
        

"""
# Calculate the cost of each subset and find the minimum
params = [0.15, 0.05, 0.001] # Typical prostate IVIM parameters. Results vary with the D* value!
costs_conditioned = [cost_function(b_values_conditioned[i], params) for i in tqdm(range(len(b_values_conditioned)))]
idx_conditioned = np.where(costs_conditioned == min(costs_conditioned))[0][0] # Find the index of the lowest cost. This index corresponds to the lowest b-value subset in the b_values subset list


# Same as above for the unconditioned list
costs_unconditioned = [cost_function(b_values_conditioned[i], params) for i in tqdm(range(len(b_values_conditioned)))]
idx_unconditioned = np.where(costs_unconditioned == min(costs_unconditioned))[0][0]

# Print results
print()
print("Conditioned optimal b-value set: {}".format(b_values_conditioned[idx_conditioned]))
print()
print("Unconditioned optimal b-value set: {}".format(b_values_unconditioned[idx_unconditioned]))
"""
unconditioned_sets = [i.subset for i in b_values_unconditioned]
conditioned_sets = [i.subset for i in b_values_conditioned]