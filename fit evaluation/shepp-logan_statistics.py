# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:45:18 2023

@author: Ivan
"""

import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class table_of_simulation_statistics:
    def __init__(self, filename, path):
        self.filename = filename
        self.path = path
        self.csv_file_to_df()
    
    def csv_file_to_df(self):
        # Get information from the filename
        self.rice_sigma = int(''.join(x for x in self.filename if x.isdigit()))
        self.parameter = self.filename[0:2].strip()
        
        # Read the csv-file and store the table in a dataframe
        self.data = pd.read_csv(os.path.join(self.path, self.filename))
        
    def change_param_name(self, new_name):
        self.parameter = new_name

path = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Shepp-Logan simuleringsdata\Linear SNR 0 50 240 800"
csv_files = os.listdir(path)

# Read the tables and sort them into lists based on the parameter
tables = [table_of_simulation_statistics(filename, path) for filename in csv_files]

f_tables = [table for table in tables if table.parameter == "f"]
D_tables = [table for table in tables if table.parameter == "D"]
D_star_tables = [table for table in tables if table.parameter == "D_"]
_ = [table.change_param_name("D*") for table in D_star_tables if table.parameter == "D_"]



def plot_maps(tables):
    # Our x-axis is bob.data["{} ground truth".format(bob.parameter)]
    # Our y-axis is bob.rice_sigma
    # Our pixel intensities are bob.data["{} bias".format(bob.parameter)]
    # and bob.data["{} std".format(bob.parameter)]
    
    y_axis = np.sort([table.rice_sigma for table in tables])
    x_axis = tables[0].data["{} ground truth".format(tables[0].parameter)]
    std_map = np.zeros((len(y_axis), len(x_axis)))
    bias_map = np.zeros((len(y_axis), len(x_axis)))

    for noise_index in range(len(y_axis)):
        noise_std_row = [table.data["{} fit std".format(table.parameter)] for table in tables if table.rice_sigma == y_axis[noise_index]]
        noise_bias_row = [-table.data["{} bias".format(table.parameter)] for table in tables if table.rice_sigma == y_axis[noise_index]]
        std_map[noise_index,:] = noise_std_row[0]
        bias_map[noise_index,:] = noise_bias_row[0]
        
    if tables[0].parameter == "D":
        x_axis_ticks = [0, 0.001, 0.002, 0.003, 0.004, 0.0047]
    else:
        x_axis_ticks = x_axis
        
    fig, axs = plt.subplots(ncols=2, figsize=(12,5))
    cbar0 = axs[0].pcolormesh(x_axis, y_axis, std_map, cmap="gray")
    axs[0].set_yscale("log")
    axs[0].set_yticks(y_axis)
    axs[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()) # Makes the y-ticks show in the log scale
    axs[0].set_xticks(x_axis_ticks)
    axs[0].set_title("Standard deviation")
    axs[0].set_ylabel("SNR at b=800", fontsize="large")
    axs[0].set_xlabel("{}".format(tables[0].parameter), fontsize="large")
    axs[0].tick_params(left=False)

    cbar1 = axs[1].pcolormesh(x_axis, y_axis, bias_map, cmap="gray")
    axs[1].set_yscale("log")
    axs[1].set_yticks(y_axis)
    axs[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()) # Makes the y-ticks show in the log scale
    axs[1].set_xticks(x_axis_ticks)
    axs[1].set_title("Bias")
    axs[1].set_ylabel("SNR at b=800", fontsize="large")
    axs[1].set_xlabel("{}".format(tables[0].parameter), fontsize="large")
    axs[1].tick_params(left=False)

    fig.colorbar(cbar0, ax=axs[0])
    fig.colorbar(cbar1, ax=axs[1])
    fig.tight_layout()
    
plot_maps(f_tables)
plot_maps(D_tables)
#plot_maps(D_star_tables)
