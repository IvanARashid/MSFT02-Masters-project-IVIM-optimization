# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:50:19 2023

@author: Ivan
"""

import numpy as np
from scipy.stats import rice

def cost_function(b_values, ivim_params, return_cost_terms=False):
    """
    

    Parameters
    ----------
    b_values : array-like
        A list of b-values to evaluate the cost for.
    ivim_params : array-like
        List of the ivim parameters to evaluate the b-values at. Should be
        ordered according to [f, Dp, Dt], where f is the perfusion fraction,
        Dp the fast diffusion, and Dt the slow diffusion.
    return_cost_terms : bool, optional
        Set to true to return the total cost, the sum factor, and the 
        individual terms f_term, Dp_term, and Dt_term. The default is False.

    Returns
    -------
    cost : float
        The cost of the b-values.
    """
    
    # Define the partial derivatives of the biexponential IVIM model
    dS_df = lambda b, f, Dp, Dt: (np.exp(-b*Dp) - np.exp(-b*Dt))
    dS_dDp = lambda b, f, Dp, Dt: f*np.exp(-b*Dp)*b
    dS_dDt = lambda b, f, Dp, Dt: (1-f)*np.exp(-b*Dt)*b
    pd_lst = [dS_df, dS_dDp, dS_dDt]
    
    # Create the Jacobian matrix where the partial derivates of the IVIM model
    # are evaluated for each b-value given the model parameters
    J = np.zeros((len(b_values), 3))
    for i in range(len(b_values)):
        for j in range(len(ivim_params)):
            J[i,j] = pd_lst[j](b_values[i], *ivim_params)
    
    # Create the Fisher information matrix by multiplying the transpose of J
    # with J
    try:
        F = np.matmul(J.T, J)
        F_inv = np.linalg.inv(F) # Invert the matrix
        
        # Determine the coefficient of variance for each parameter using the
        # diagonal values of the inverted Fisher information matrix
        f_term = np.sqrt(F_inv[0,0])/ivim_params[0]
        Dp_term = np.sqrt(F_inv[1,1])/ivim_params[1]
        Dt_term = np.sqrt(F_inv[2,2])/ivim_params[2]
        
        # Determines the sum factor which multiplies the cost with the total sum
        # of the number of occurences for each b-value 
        sum_factor = 0
        for i in range(len(b_values)):
            for j in range(len(b_values)):
                if b_values[i] == b_values[j]:
                    sum_factor += 1
        
        cost = np.sqrt(sum_factor * (f_term+Dp_term+Dt_term))
    except:
        S0_term = 1000
        f_term = 1000
        Dp_term = 1000
        Dt_term = 1000
        cost = 1e6
    
    if return_cost_terms:
        return cost, sum_factor, f_term, Dp_term, Dt_term
    else:
        return cost
    


def cost_function_S0(b_values, ivim_params, return_cost_terms=False):
    """
    

    Parameters
    ----------
    b_values : array-like
        A list of b-values to evaluate the cost for.
    ivim_params : array-like
        List of the ivim parameters to evaluate the b-values at. Should be
        ordered according to [f, Dp, Dt], where f is the perfusion fraction,
        Dp the fast diffusion, and Dt the slow diffusion.
    return_cost_terms : bool, optional
        Set to true to return the total cost, the sum factor, and the 
        individual terms f_term, Dp_term, and Dt_term. The default is False.

    Returns
    -------
    cost : float
        The cost of the b-values.
    """
    S_bi = lambda b, S0, f, Dp, Dt: S0*((1-f)*np.exp(-b*Dt) + f*np.exp(-b*Dp))
    # Define the partial derivatives of the biexponential IVIM model
    dS_dS0 = lambda b, S0, f, Dp, Dt: (1-f)*np.exp(-b*Dt) + f*np.exp(-b*Dp)
    dS_df = lambda b, S0, f, Dp, Dt: S0*(np.exp(-b*Dp) - np.exp(-b*Dt))
    dS_dDp = lambda b, S0, f, Dp, Dt: -S0*f*np.exp(-b*Dp)*b
    dS_dDt = lambda b, S0, f, Dp, Dt: -S0*(1-f)*np.exp(-b*Dt)*b
    pd_lst = [dS_dS0, dS_df, dS_dDp, dS_dDt]
    
    # We only accept integer b-values in steps of 10
    #b_values = [round(int(b_value), -1) for b_value in b_values]
    
    # Create the Jacobian matrix where the partial derivates of the IVIM model
    # are evaluated for each b-value given the model parameters
    J = np.zeros((len(b_values), 4))
    for i in range(len(b_values)):
        for j in range(len(ivim_params)):
            J[i,j] = pd_lst[j](b_values[i], *ivim_params)
    
    nex_list = []
    for i in range(len(b_values)):
        nex_i = 0
        for j in range(len(b_values)):
            if round(b_values[i],-1) == round(b_values[j],-1):
                nex_i += 1
        nex_list.append(nex_i)
    
    # Cov matrix is 1/sigma**2
    # Sigma has been shown to depend on 1/sqrt(nex)
    # Therefore, 1/(1/sqrt(nex))**2 = nex => the diagonal of the covariance matrix is simply the nex for each b-value
    cov_matrix = np.diag(nex_list)
    
    # Create the Fisher information matrix by multiplying the transpose of J
    # with J
    try:
        #F = np.matmul(J.T, J)
        F = np.matmul(np.matmul(J.T,cov_matrix), J) # Create the fisher matrix, including a covariance matrix
        F_inv = np.linalg.inv(F) # Invert the matrix
        
        # Determine the coefficient of variance for each parameter using the
        # diagonal values of the inverted Fisher information matrix
        S0_term = np.sqrt(F_inv[0,0])/ivim_params[0]
        f_term = np.sqrt(F_inv[1,1])/ivim_params[1]
        Dp_term = np.sqrt(F_inv[2,2])/ivim_params[2]
        Dt_term = np.sqrt(F_inv[3,3])/ivim_params[3]
        
        # Determines the sum factor which multiplies the cost with the total sum
        # of the number of occurences for each b-value 
        # Removed, this should be taken care of by the matrix multiplication
        """
        sum_factor = 0
        for i in range(len(b_values)):
            for j in range(len(b_values)):
                if b_values[i] == b_values[j]:
                    sum_factor += 1
        """
        
        #cost = np.sqrt(sum_factor * (S0_term+f_term+Dp_term+Dt_term))
        #cost = np.sqrt(S0_term+f_term+Dp_term+Dt_term)
        cost = S0_term + f_term + Dp_term + Dt_term
        #cost = f_term + Dp_term+ Dt_term
    except:
        S0_term = 1000
        f_term = 1000
        Dp_term = 1000
        Dt_term = 1000
        cost = 1e6
    
    if return_cost_terms:
        return cost, f_term, Dp_term, Dt_term
    else:
        return cost

def cost_function_S0_no_covar(b_values, ivim_params, return_cost_terms=False):
    """
    

    Parameters
    ----------
    b_values : array-like
        A list of b-values to evaluate the cost for.
    ivim_params : array-like
        List of the ivim parameters to evaluate the b-values at. Should be
        ordered according to [f, Dp, Dt], where f is the perfusion fraction,
        Dp the fast diffusion, and Dt the slow diffusion.
    return_cost_terms : bool, optional
        Set to true to return the total cost, the sum factor, and the 
        individual terms f_term, Dp_term, and Dt_term. The default is False.

    Returns
    -------
    cost : float
        The cost of the b-values.
    """
    S_bi = lambda b, S0, f, Dp, Dt: S0*((1-f)*np.exp(-b*Dt) + f*np.exp(-b*Dp))
    # Define the partial derivatives of the biexponential IVIM model
    dS_dS0 = lambda b, S0, f, Dp, Dt: (1-f)*np.exp(-b*Dt) + f*np.exp(-b*Dp)
    dS_df = lambda b, S0, f, Dp, Dt: S0*(np.exp(-b*Dp) - np.exp(-b*Dt))
    dS_dDp = lambda b, S0, f, Dp, Dt: -S0*f*np.exp(-b*Dp)*b
    dS_dDt = lambda b, S0, f, Dp, Dt: -S0*(1-f)*np.exp(-b*Dt)*b
    pd_lst = [dS_dS0, dS_df, dS_dDp, dS_dDt]
    
    # We only accept integer b-values in steps of 10
    #b_values = [round(int(b_value), -1) for b_value in b_values]
    
    # Create the Jacobian matrix where the partial derivates of the IVIM model
    # are evaluated for each b-value given the model parameters
    J = np.zeros((len(b_values), 4))
    for i in range(len(b_values)):
        for j in range(len(ivim_params)):
            J[i,j] = pd_lst[j](b_values[i], *ivim_params)
    
    nex_list = []
    for i in range(len(b_values)):
        nex_i = 0
        for j in range(len(b_values)):
            if b_values[i] == b_values[j]:
                nex_i += 1
        nex_list.append(nex_i)
    
    # Cov matrix is 1/sigma**2
    # Sigma has been shown to depend on 1/sqrt(nex)
    # Therefore, 1/(1/sqrt(nex))**2 = nex => the diagonal of the covariance matrix is simply the nex for each b-value
    cov_matrix = np.diag(nex_list)
    
    # Create the Fisher information matrix by multiplying the transpose of J
    # with J
    try:
        F = np.matmul(J.T, J)
        #F = np.matmul(np.matmul(J.T,cov_matrix), J) # Create the fisher matrix, including a covariance matrix
        F_inv = np.linalg.inv(F) # Invert the matrix
        
        # Determine the coefficient of variance for each parameter using the
        # diagonal values of the inverted Fisher information matrix
        S0_term = np.sqrt(F_inv[0,0])/ivim_params[0]
        f_term = np.sqrt(F_inv[1,1])/ivim_params[1]
        Dp_term = np.sqrt(F_inv[2,2])/ivim_params[2]
        Dt_term = np.sqrt(F_inv[3,3])/ivim_params[3]
        
        # Determines the sum factor which multiplies the cost with the total sum
        # of the number of occurences for each b-value 
        # Removed, this should be taken care of by the matrix multiplication
        
        sum_factor = 0
        for i in range(len(b_values)):
            for j in range(len(b_values)):
                if b_values[i] == b_values[j]:
                    sum_factor += 1
        
        
        #cost = np.sqrt(sum_factor * (S0_term+f_term+Dp_term+Dt_term))
        #cost = np.sqrt(S0_term+f_term+Dp_term+Dt_term)
        cost = S0_term + f_term + Dp_term + Dt_term
        #cost = f_term + Dp_term+ Dt_term
    except:
        S0_term = 1000
        f_term = 1000
        Dp_term = 1000
        Dt_term = 1000
        cost = 1e6
    
    if return_cost_terms:
        return cost, f_term, Dp_term, Dt_term
    else:
        return cost
    

"""
params = [1000, 0.1, 0.02, 0.001]
b_values = [0, 50, 240, 250, 260, 790, 800, 810]
b_values2 = [0, 50, 250, 250, 250, 800, 800, 800]
res1 = cost_function_S0(b_values, params)
res2 = cost_function_S0_no_covar(b_values, params)
print(res1)
print(res2)
print()
print(cost_function_S0(b_values2, params))
print(cost_function_S0_no_covar(b_values2, params))
"""