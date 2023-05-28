# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:27:59 2023

@author: Ivan
"""
import numpy as np
from tqdm import tqdm
from dipy.reconst.ivim import IvimModel
from dipy.core.gradients import gradient_table
import topopro
from scipy.optimize import curve_fit

def ivim_signal(b, f, D_star, D, S0=1):
    return S0*(f*np.exp(-b*D_star) + (1-f)*np.exp(-b*D))

n = 10 # Number of voxels in each dimension
b_values = [0, 50, 240, 800] # The b-values for the signal calculation

# Creates the gradients
f = np.linspace(0.01, 0.5, n)
D_star = np.linspace(0.004, 0.01)
D = np.linspace(0.0001, 0.001)

# Create a table output for the parameter values

# Create the phantom volume, with each voxel having a unique set of parameter values
f, D_star, D = np.meshgrid(f, D_star, D)

# Calculate the IVIM-signal for each voxel
signal = np.zeros((n, n, n, len(b_values)))
for f_index in tqdm(range(n)):
    for D_star_index in range(n):
        for D_index in range(n):
            for b_value_index in range(len(b_values)):
                signal[f_index, D_star_index, D_index, b_value_index] = ivim_signal(b_values[b_value_index],
                                                                                    f[f_index, D_star_index, D_index],
                                                                                    D_star[f_index, D_star_index, D_index],
                                                                                    D[f_index, D_star_index, D_index])

def fit_VarPro(signal_volume, bvals=b_values):
    # Gradient table
    bvec = np.zeros((bvals.size, 3))
    bvec[:,2] = 1
    gtab = gradient_table(bvals, bvec, b0_threshold=0)
    
    # Create model
    bounds_lower = [0, 0, 0]
    bounds_upper = [1, 0.1, 0.004]
    ivimmodel = IvimModel(gtab, fit_method="VarPro", bounds=(bounds_lower, bounds_upper))
    
    ivimfit = ivimmodel.fit(signal_volume)
    
    S0_fit = ivimfit.model_params[:,:,:,0]
    f_fit = ivimfit.model_params[:,:,:,1]
    D_star_fit = ivimfit.model_params[:,:,:,2]
    D_fit = ivimfit.model_params[:,:,:,3]
    
    return S0_fit, f_fit, D_star_fit, D_fit

def fit_TopoPro(signal_volume, bvals=b_values):
    # Gradient table
    bvec = np.zeros((bvals.size, 3))
    bvec[:,2] = 1
    gtab = gradient_table(bvals, bvec, b0_threshold=0)
    
    # Create model
    bounds_lower = [0, 0, 0]
    bounds_upper = [1, 0.1, 0.004]
    ivimmodel = IvimModel(gtab, fit_method="VarPro", bounds=(bounds_lower, bounds_upper))
    
    ivimfit = ivimmodel.fit(signal_volume)
    
    S0_fit = ivimfit.model_params[:,:,:,0]
    f_fit = ivimfit.model_params[:,:,:,1]
    D_star_fit = ivimfit.model_params[:,:,:,2]
    D_fit = ivimfit.model_params[:,:,:,3]
    
    return S0_fit, f_fit, D_star_fit, D_fit

def fit_linear(signal_volume, bvals=b_values, b_threshold=200):
    # Create the result images
    f = np.empty(signal_volume.shape[0:-1])
    D = np.empty(signal_volume.shape[0:-1])
    S0 = np.empty(signal_volume.shape[0:-1])

    # Function for S0
    def s0_voxel_value(s_b0, f):
        s0 = s_b0/(1-f)
        s0 = np.nan_to_num(s0)
        return s0

    # The model to be fit
    def func(b, adc, intercept):
        res = -b*adc + intercept
        return res
    
    bval_index_start = np.where(bvals >= b_threshold)[0][0]
    xdata = [b for b in bvals if b >= b_threshold]

    for row in tqdm(range(signal_volume.shape[0])):
        for col in range(signal_volume.shape[1]):
            for slice in range(signal_volume.shape[2]):
                # Fit the data to the function
                ydata = np.log(signal_volume[row, col, slice, bval_index_start:]/signal_volume[row, col, slice, 0])
                ydata = np.nan_to_num(ydata)
                res = curve_fit(func, xdata, ydata, bounds=([0, -1], [0.004,0]))[0]
                
                # Assign the fitted parameters to their voxels in the parameter maps
                f[row, col, slice] = res[1]
                D[row, col, slice] = res[0]
                
                # Calculate S0
                S0[row, col, slice] = s0_voxel_value(signal_volume[row, col, slice, 0], res[1])
                
    return S0, f, D

def plot_bias(param_map, ground_truth, D_slice):
    bias = param_map[:,:,D_slice] - ground_truth[:,:,D_slice]
