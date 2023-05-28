# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:20:33 2023

@author: Ivan

Script that plots the standard deviation data of the 6D noised signal simulation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def filter_nans(table, column):
    filtered_lst = [table[column][i] for i in range(len(table[column])) if not np.isnan(table[column][i])]
    return filtered_lst

path = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\Standard deviation"
filename = "Linear_test.csv"

table = pd.read_csv(os.path.join(path, filename))

# Filter out the nans for some of the columns
b_values = filter_nans(table, "b-values")
SNRs = filter_nans(table, "SNR")
fs = filter_nans(table, "f")
D_stars = filter_nans(table, "D*")
Ds = filter_nans(table, "D")

# Number of total elements per SNR level is the total number of rows divided by the number of SNRs
elements_per_SNR = int(np.round(len(table["Channel 0"])/len(SNRs)))
element_ranges = [range(i*elements_per_SNR, (i+1)*elements_per_SNR) for i in range(len(SNRs))]

# The number of elements per axis per SNR level is (# elements)^(1/3)
n = int(np.round((len(table["Channel 0"])/len(SNRs))**(1/3)))

# Dimensions i, j, k, snr
# Create array that holds all the data
standard_deviations = np.zeros((n,n,n,len(SNRs)))
for i in range(len(table["Channel 0"])):
    standard_deviations[table["Row (j)"][i], table["Column (i)"][i], table["Slice (k)"][i], table["Channel 0"][i]] = table["Standard Deviation"][i]


