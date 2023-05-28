import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.colors as mcolors
import matplotlib
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.lines import Line2D
import seaborn as sns
import scienceplots
import cmasher as cmr

plt.style.use("default")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['text.usetex'] = True # Prevents font change when italicized

    
def load_tables(path, quant):
    data = pd.read_csv(os.path.join(path, f"{quant}.csv"))
    return data

class FitMethod:
    def __init__(self, path, fit_name):
        self.path = path
        self.fit_name = fit_name
         
        quants = ["rmse", "bias", "std"]
        self.f_rmse, self.f_bias, self.f_std = [self._load_data("f", quant) for quant in quants]
        self.f_rmse_total = np.sum(self.f_rmse, axis=(0, 1, 2)) # f_rmse is 4D, we sum the rmse for each SNR-level
        self.f_bias_sq_total = np.sum(np.sqrt(np.square(self.f_bias)), axis=(0, 1, 2)) # f_bias is 3D
        self.f_std_sq_total = np.sum(np.sqrt(np.square(self.f_std)), axis=(0, 1, 2)) # f_std is 4D, we sum the rmse for each SNR-level
        
        self.D_rmse, self.D_bias, self.D_std = [self._load_data("D", quant) for quant in quants]
        self.D_rmse_total = np.sum(self.D_rmse, axis=(0, 1, 2))
        self.D_bias_sq_total = np.sum(np.sqrt(np.square(self.D_bias)), axis=(0, 1, 2)) # D_bias is 3D
        self.D_std_sq_total = np.sum(np.sqrt(np.square(self.D_std)), axis=(0, 1, 2)) # D_std is 4D, we sum the rmse for each SNR-level
        
        if self.fit_name.lower() not in ["linear", "sivim"]:
            self.D_star_rmse, self.D_star_bias, self.D_star_std = [self._load_data("D_star", quant) for quant in quants]
            self.D_star_rmse_total = np.sum(self.D_star_rmse, axis=(0, 1, 2))
            self.D_star_bias_sq_total = np.sum(np.sqrt(np.square(self.D_star_bias)), axis=(0, 1, 2)) # D_star_bias is 3D
            self.D_star_std_sq_total = np.sum(np.sqrt(np.square(self.D_star_std)), axis=(0, 1, 2)) # D_std is 4D, we sum the rmse for each SNR-level
            
        
    def _load_data(self, parameter, quant):
        """
        Loads the data from the csv file

        Args:
            parameter (string): f, D_star, or D.
            quant (string): bias, rmse, or std.
        """
        data = np.load(os.path.join(self.path, f"{self.fit_name.lower()} {parameter} {quant}.npy"))
        return data
    


path_standard = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\b[0, 50, 240, 800] SNR[3, 10, 30] samples 100"
path_low = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\b[0, 25, 50, 240, 800] SNR[3, 10, 30] samples 100"
path_intermediate = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\b[0, 50, 145, 240, 800] SNR[3, 10, 30] samples 100"
path_high = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\b[0, 50, 240, 520, 800] SNR[3, 10, 30] samples 100"
path_all = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\b[0, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 200, 240, 400, 520, 800] SNR[3, 10, 30] samples 100"
path_opt_no_covar = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\multinexb[0, 15, 30, 50, 240, 800] SNR[3, 10, 30] samples 100"
path_opt_covar = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\multinexb[0, 20, 40, 60, 280, 470, 800] SNR[3, 10, 30] samples 100"
path_generic = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\multinexb[0, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800] SNR[3, 10, 30] samples 100"

fit_methods = ["Linear", "VarPro", "TopoPro"]
parameters = ["f", "D*", "D"]
bvals_standard = [0, 50, 240, 800]
bvals_low = [0, 25, 50, 240, 800]
bvals_intermediate = [0, 50, 145, 240, 800]
bvals_high = [0, 50, 240, 520, 800]
bvals_all = [0, 800]
bvals = [bvals_standard, bvals_low, bvals_intermediate, bvals_high, bvals_all]

SNR_table = load_tables(path_standard, "SNR_table")
b_values = load_tables(path_standard, "b_table")
parameter_table = load_tables(path_standard, "parameter_table")

    
def plot_rmse_grid(data, parameter, D_idx=3, parameter_table=parameter_table, SNR_table=SNR_table, normscale="linear", global_colorbar_norm=False, cmap="inferno", fontsize=12, save_name=None):
    plt.style.use("default")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True # Prevents font change when italicized
    
    figsize= (6.27, 9.69)
    if parameter == "D*":
        data = data[2:] # Remove the linear  and sIVIM fits, since it does not contain any D* data
        figsize = (6.27,6.46)
    
    # Set up fig and img grid
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(len(data), len(SNR_table["SNR 0"])),
                     axes_pad=0.1,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size=.3,
                     cbar_pad=0.15,
                     aspect=False, # Needed if grid not square
                     direction="row")
    
    noise_idx_list = list(range(len(SNR_table["SNR 0"]+1)))*len(data) # List of noise idx for every occurence of data
    data_expanded = [ele for ele in data for i in range(len(SNR_table["SNR 0"]))] # Creates a sorted list with repeated values of data
    data_expanded = []
    if parameter == "f":
        data_expanded = [ele.f_rmse for ele in data for i in range(len(SNR_table["SNR 0"]))] # Creates a sorted list with repeated values of data
    elif parameter == "D*":
        data_expanded = [ele.D_star_rmse for ele in data for i in range(len(SNR_table["SNR 0"]))] # Creates a sorted list with repeated values of data
    elif parameter == "D":
        data_expanded = [ele.D_rmse for ele in data for i in range(len(SNR_table["SNR 0"]))] # Creates a sorted list with repeated values of data
    
    # Initiate colormap normalisation
    vmin = 0.99
    vmax = 0
    # Set to global_colorbar_norm to True if the vmin and vmax should be set to the max and min of all D_idx's
    if global_colorbar_norm:
        for ax_idx in range(len(data_expanded)):
            if np.max(data_expanded[ax_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx])
            if np.min(data_expanded[ax_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx])
    else:
        for ax_idx in range(len(data_expanded)):
            if np.max(data_expanded[ax_idx][:,:,D_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx][:,:,D_idx])
            if np.min(data_expanded[ax_idx][:,:,D_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx][:,:,D_idx])
    
    if normscale == "linear":
        norm=colors.Normalize(vmin=vmin, vmax=vmax)
    elif normscale == "log":
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm=colors.Normalize(vmin=vmin, vmax=vmax)
        
    
    # Lists of indexes of the axes where labels shall be placed
    title_labels = list(range(len(SNR_table["SNR 0"])))
    y_labels = [len(SNR_table["SNR 0"])*i for i in range(len(data))]
    ylabels = ["Linear", "sIVIM", "Subtracted", "Segmented", "MIX", "TopoPro"]
    x_labels = list(range(len(data_expanded)-len(SNR_table["SNR 0"]), len(data_expanded)))
    if parameter == "D*":
        ylabels = ylabels[2:]
    
    # Fill the image grid
    #yscale = [parameter_table["D*"][i]/parameter_table["D"][D_idx] for i in range(len(parameter_table["D*"]))]
    yscale = parameter_table["D*"]*1000
    for ax_idx in range(len(grid)):
        im = grid[ax_idx].pcolormesh(parameter_table["f"]*100, yscale, data_expanded[ax_idx][:,:,D_idx,noise_idx_list[ax_idx]], cmap=cmap, norm=norm)
        
        """
        if np.max(data_expanded[ax_idx].D_rmse[:,:,D_idx, noise_idx_list[ax_idx]]) > vmax:
            vmax = np.max(data_expanded[ax_idx].D_rmse[:,:,D_idx, noise_idx_list[ax_idx]])
        if np.min(data_expanded[ax_idx].D_rmse[:,:,D_idx, noise_idx_list[ax_idx]]) < vmin:
            vmin = np.min(data_expanded[ax_idx].D_rmse[:,:,D_idx, noise_idx_list[ax_idx]])
        """
            
        if ax_idx in title_labels:
            grid[ax_idx].set_title(f"SNR = {SNR_table['SNR 0'][noise_idx_list[ax_idx]]:.0f}")
        if ax_idx in y_labels:
            grid[ax_idx].set_ylabel(f"{ylabels[int(np.ceil((ax_idx+1)/len(SNR_table['SNR 0'])-1))]} \n $D^*$ [µm$^2$/ms]", fontsize="large")
        if ax_idx in x_labels:
            grid[ax_idx].set_xlabel("$f$ [\%]", fontsize="large")
            grid[ax_idx].set_xticks([10, 30, 50])
            grid[ax_idx].tick_params(labelsize="large")
    
    # Colorbar ticks 
    ticks = list(np.linspace(vmin, vmax, 9))
    #ticks = [tick if round(tick*1000, 1) != 0 else 0 for tick in ticks]
    
    if parameter == "f":
        unit = "[\%]"
    else:
        unit = "[µm$^2$/ms]"
    
    if parameter == "D*":
        label_parameter = "D^*"
    else:
        label_parameter = parameter    
    
    cblabel = f"${label_parameter}$ RMSE {unit} for $D$ = {parameter_table['D'][D_idx]*1000:.1f} µm$^2$/ms"
    
    grid[-1].cax.cla()
    cbar = matplotlib.colorbar.Colorbar(grid[-1].cax,im, label=cblabel)
    # Set colorbar ticklabels
    if parameter == "D*":
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([int(round(val*1000, 0)) for val in ticks], fontsize=12)
    elif parameter == "D":
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([round(val*1000, 1) for val in ticks], fontsize=12)
    elif parameter == "f":
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([int(np.ceil(val*100)) for val in ticks], fontsize=12)
        
    fig.tight_layout()
    if save_name:
        fig.savefig(f"{save_name}.pdf")
    
def plot_f_bias_grid_vs_bvals(data, parameter, bvals=bvals, D_idx=3, parameter_table=parameter_table, normscale="linear", global_colorbar_norm=False, b_sets=None, save_name=False, ylabels=None, titles=None, figsize=None, cbar_size=.3, crop_Dstar=True, filetype="pdf", fontsize=12):
    plt.style.use("default")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True # Prevents font change when italicized
    """
    Plots a bias map for different fitting algorithms and different b-value sets.

    Args:
        data (list): list of lists. First level is fitting algorithms; 2nd level is a list of FitMethod objects for different b-value simulations.
        parameter (string): The parameter to plot the heat maps for. Set to "f", or "D*", or "D".
        bvals (list, optional): List of list of b-value sets. Defaults to bvals.
        D_idx (int, optional): The D_idx to show the heat maps for. Defaults to 3.
        parameter_table (DataFrame, optional): DataFrame object of read parameter_table from simulations. Defaults to parameter_table.
        normscale (str, optional): Change to log-scaled colormap. May not work due to zero-valued elements. Defaults to "linear".
        global_colorbar_norm (bool, optional): Whether to normalsie the colour map to the values of the D_idx-slice, or the entire volume. Defaults to False.
    """
    if parameter == "D*":
        if crop_Dstar:
            data = data[2:] # Remove the linear  and sIVIM fits, since it does not contain any D* data
            
        if figsize:
            figsize = figsize
        else:
            figsize= (6.27, 5.33)
            
    if figsize:
        figsize = figsize
    else:
        figsize= (6.27, 8)
    
    # Set up fig and img grid
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(len(data), len(data[0])),
                     axes_pad=0.1,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     #cbar_size="4%",
                     cbar_size=cbar_size,
                     cbar_pad=0.15,
                     aspect=False, # Needed if grid not square
                     direction="row")
    
    # Create the data_expanded list that is looped over along the axes when creating the plots
    data_expanded = []
    if parameter == "f":
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx])):
                data_expanded.append(data[dataset_idx][b_set_idx].f_bias)# - data[dataset_idx][0].f_bias)
    elif parameter == "D*":
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx])):
                data_expanded.append(data[dataset_idx][b_set_idx].D_star_bias)# - data[dataset_idx][0].D_star_bias)
    elif parameter == "D":
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx])):
                data_expanded.append(data[dataset_idx][b_set_idx].D_bias)# - data[dataset_idx][0].D_bias)
    
    # Initiate colormap normalisation
    vmin = 0.99
    vmax = 0
    # Set to global_colorbar_norm to True if the vmin and vmax should be set to the max and min of all D_idx's
    if parameter == "D*":
        exclude = 8
        #exclude = 0
        #vmax = 10e-3
    else:
        exclude = 0
    if global_colorbar_norm:
        for ax_idx in range(len(data_expanded)-exclude):
            if np.max(data_expanded[ax_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx])
            if np.min(data_expanded[ax_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx])
    else:
        for ax_idx in range(len(data_expanded)-exclude):
            if np.max(data_expanded[ax_idx][:,:,D_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx][:,:,D_idx])
            if np.min(data_expanded[ax_idx][:,:,D_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx][:,:,D_idx])
                
            
    if parameter == "D*":
        vmin = -vmax
    
    if parameter == "D":
        vmin = -0.3e-3
    
    v_ext = np.max([np.abs(vmin), np.abs(vmax)])
    # LogNorm is not an appropriate scale for diverging maps, should be removed or changed
    if normscale == "linear":            
        norm=colors.TwoSlopeNorm(vmin=vmin-1e-5, vcenter=0., vmax=vmax+1e-5)
        #norm=colors.TwoSlopeNorm(vmin=-v_ext, vcenter=0., vmax=v_ext)
    elif normscale == "log":
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm=colors.TwoSlopeNorm(vmin=vmin-1e-5, vcenter=0., vmax=vmax+1e-5)
               
    
    # Lists of indexes of the axes where labels shall be places
    title_labels = list(range(len(data[0])))
    y_labels = [len(data[0])*i for i in range(len(data))]
    x_labels = list(range(len(data_expanded)-len(data[0]), len(data_expanded))) 
    
    if titles:
        titles = titles
    else:
        titles = ["Opt no covar", "Opt covar", "Generic"]*len(data)
    
    if ylabels:
        ylabels = ylabels
    else:
        ylabels = ["Linear", "sIVIM", "Subtracted", "Segmented", "MIX", "TopoPro"]
    if parameter == "D*" and crop_Dstar:
        ylabels = ylabels[2:]
        
    if b_sets != None:
        titles = b_sets
        
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        #def __call__(self, value, clip=None):
            ## I'm ignoring masked values and all kinds of edge cases to make a
            ## simple example...
            #x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            #return np.ma.masked_array(np.interp(value, x, y))
        
        def __call__(self, value, clip=None):
            v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
            x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))
        
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    
    cmap = "coolwarm"
    # Fill the image grid
    for ax_idx in range(len(grid)):
        im = grid[ax_idx].pcolormesh(parameter_table["f"]*100, parameter_table["D*"]*1000, data_expanded[ax_idx][:,:,D_idx], cmap=cmap, norm=norm)
        
        if ax_idx in title_labels:
            grid[ax_idx].set_title(f"{titles[ax_idx]}", fontsize=fontsize)
        if ax_idx in y_labels:
            y_label_idx = ylabels[int(np.ceil((ax_idx+1)/len(data[0])-1))]
            unit = "µm $^2$/ms"
            grid[ax_idx].set_ylabel(f"{y_label_idx} \n $D^*$ [{unit}]", fontsize=fontsize)
            grid[ax_idx].tick_params(labelsize=fontsize)
        if ax_idx in x_labels:
            grid[ax_idx].set_xticks([10, 30, 50])
            grid[ax_idx].set_xlabel("$f$ [\%]", fontsize=fontsize)
            grid[ax_idx].tick_params(labelsize=fontsize)
    
    m0=vmin           # colorbar min value
    m9=vmax             # colorbar max value
    
    # Colorbar ticks 
    ticks_lower = list(np.linspace(vmin, 0, 4))
    ticks_upper = list(np.linspace(0, vmax, 4))
    ticks = ticks_lower + ticks_upper
    ticks = [tick if round(tick*1000, 1) != 0 else 0 for tick in ticks]
    
    if parameter == "f":
        unit = "[\%]"
    else:
        unit = "[µm$^2$/ms]"
        
    if parameter == "D*":
        label_parameter = "D^*"
    else:
        label_parameter = parameter
    
    cblabel = f"${label_parameter}$ bias {unit} for $D$ = {parameter_table['D'][D_idx]*1000:.1f} µm$^2$/ms"
    
    
    grid[-1].cax.cla()
    cbar = matplotlib.colorbar.Colorbar(grid[-1].cax, im, label=cblabel, ticks=ticks)
    cbar.set_label(cblabel, size=fontsize)
    # Set colorbar ticklabels
    if parameter == "D*":
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([int(round(val*1000, 0)) for val in ticks], fontsize=fontsize)
    elif parameter == "D":
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([round(val*1000, 1) for val in ticks], fontsize=fontsize)
    elif parameter == "f":
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([int(np.ceil(val*100)) for val in ticks], fontsize=fontsize)
    
    #fig.subplots_adjust(bottom=0.25)    
    fig.tight_layout()
    #fig.draw_without_rendering()
    if save_name:
        fig.savefig(f"{save_name}.{filetype}", bbox_inches="tight", dpi=300)
    
def plot_f_bias_grid_diff_vs_bvals(data, parameter, bvals=bvals, D_idx=3, parameter_table=parameter_table, normscale="linear", global_colorbar_norm=False):
    plt.style.use("default")
    plt.rcParams["font.family"] = "Times New Roman"
    """
    Plots a bias map for different fitting algorithms and different b-value sets.

    Args:
        data (list): list of lists. First level is fitting algorithms; 2nd level is a list of FitMethod objects for different b-value simulations.
        parameter (string): The parameter to plot the heat maps for. Set to "f", or "D*", or "D".
        bvals (list, optional): List of list of b-value sets. Defaults to bvals.
        D_idx (int, optional): The D_idx to show the heat maps for. Defaults to 3.
        parameter_table (DataFrame, optional): DataFrame object of read parameter_table from simulations. Defaults to parameter_table.
        normscale (str, optional): Change to log-scaled colormap. May not work due to zero-valued elements. Defaults to "linear".
        global_colorbar_norm (bool, optional): Whether to normalsie the colour map to the values of the D_idx-slice, or the entire volume. Defaults to False.
    """
    if parameter == "D*":
        data = data[2:] # Remove the linear  and sIVIM fits, since it does not contain any D* data
    
    # Set up fig and img grid
    fig = plt.figure(figsize=(8,9))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(len(data), len(data[0][1:])),
                     axes_pad=0.1,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="4%",
                     cbar_pad=0.15,
                     aspect=False, # Needed if grid not square
                     direction="row")
    
    # Create the data_expanded list that is looped over along the axes when creating the plots
    data_expanded = []
    if parameter == "f":
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx][1:])):
                data_expanded.append(data[dataset_idx][b_set_idx+1].f_bias - data[dataset_idx][0].f_bias)
    elif parameter == "D*":
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx][1:])):
                data_expanded.append(data[dataset_idx][b_set_idx+1].D_star_bias - data[dataset_idx][0].D_star_bias)
    elif parameter == "D":
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx][1:])):
                data_expanded.append(data[dataset_idx][b_set_idx+1].D_bias - data[dataset_idx][0].D_bias)
    
    # Initiate colormap normalisation
    vmin = 0.99
    vmax = 0
    # Set to global_colorbar_norm to True if the vmin and vmax should be set to the max and min of all D_idx's
    if parameter == "D*":
        exclude = 6
    else:
        exclude = 6
    if global_colorbar_norm:
        for ax_idx in range(len(data_expanded)-exclude):
            if np.max(data_expanded[ax_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx])
            if np.min(data_expanded[ax_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx])
    else:
        for ax_idx in range(len(data_expanded)-exclude):
            if np.max(data_expanded[ax_idx][:,:,D_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx][:,:,D_idx])
            if np.min(data_expanded[ax_idx][:,:,D_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx][:,:,D_idx])
    
    # LogNorm is not an appropriate scale for diverging maps, should be removed or changed
    if normscale == "linear":            
        norm=colors.TwoSlopeNorm(vmin=vmin-1e-5, vcenter=0., vmax=vmax+1e-5)
    elif normscale == "log":
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm=colors.TwoSlopeNorm(vmin=vmin-1e-5, vcenter=0., vmax=vmax+1e-5)
               
    
    # Lists of indexes of the axes where labels shall be places
    title_labels = list(range(len(data[0][1:])))
    y_labels = [3*i for i in range(len(data))]
    x_labels = list(range(len(data_expanded)-len(data[0][1:]), len(data_expanded))) 
    titles = ["+ low", "+ intermediate", "+ high"]*len(data)
    ylabels = ["Linear", "sIVIM", "Segmented", "TRR", "VarPro", "TopoPro"]
    if parameter == "D*":
        ylabels = ylabels[2:]
    
    # Fill the image grid
    for ax_idx in range(len(grid)):
        im = grid[ax_idx].pcolormesh(parameter_table["f"], parameter_table["D*"], data_expanded[ax_idx][:,:,D_idx], cmap="coolwarm", norm=norm)
        
            
        if ax_idx in title_labels:
            grid[ax_idx].set_title(f"{titles[ax_idx]}")
        if ax_idx in y_labels:
            grid[ax_idx].set_ylabel(f"{ylabels[int(np.ceil((ax_idx+1)/len(data[0][1:])-1))]} $D^*$")
        if ax_idx in x_labels:
            grid[ax_idx].set_xlabel("$f$", fontsize="large")
    
    m0=vmin           # colorbar min value
    m9=vmax             # colorbar max value
    
    # Colorbar ticks 
    ticks_lower = list(np.linspace(vmin, 0, 4))
    ticks_upper = list(np.linspace(0, vmax, 4))
    ticks = ticks_lower + ticks_upper
    ticks = [tick if round(tick*1000, 1) != 0 else 0 for tick in ticks]
    
    if parameter == "f":
        unit = "[%]"
    else:
        unit = "[µm$^2$/ms]"
    cblabel = f"{parameter} change in bias {unit} for D = {parameter_table['D'][D_idx]*1000:.1f} µm$^2$/ms"
    
    grid[-1].cax.cla()
    cbar = matplotlib.colorbar.Colorbar(grid[-1].cax , im, label=cblabel, ticks=ticks)
    # Set colorbar ticklabels
    if parameter != "f":
        cbar.ax.set_yticklabels([round(val*1000, 2) for val in ticks])
    else:
        cbar.ax.set_yticklabels([round(val*100, 2) for val in ticks])
        
    fig.tight_layout()

def plot_rmse_grid_vs_bvals(data, parameter, SNR_idx=0, bvals=bvals, D_idx=3, parameter_table=parameter_table, normscale="linear", global_colorbar_norm=False, save_name=False):
    plt.style.use("default")
    plt.rcParams["font.family"] = "Times New Roman"
    """
    Plots a bias map for different fitting algorithms and different b-value sets.

    Args:
        data (list): list of lists. First level is fitting algorithms; 2nd level is a list of FitMethod objects for different b-value simulations.
        parameter (string): The parameter to plot the heat maps for. Set to "f", or "D*", or "D".
        bvals (list, optional): List of list of b-value sets. Defaults to bvals.
        D_idx (int, optional): The D_idx to show the heat maps for. Defaults to 3.
        parameter_table (DataFrame, optional): DataFrame object of read parameter_table from simulations. Defaults to parameter_table.
        normscale (str, optional): Change to log-scaled colormap. May not work due to zero-valued elements. Defaults to "linear".
        global_colorbar_norm (bool, optional): Whether to normalsie the colour map to the values of the D_idx-slice, or the entire volume. Defaults to False.
    """
    if parameter == "D*":
        data = data[2:] # Remove the linear  and sIVIM fits, since it does not contain any D* data
    
    # Set up fig and img grid
    fig = plt.figure(figsize=(8,9))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(len(data), len(data[0])),
                     axes_pad=0.1,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="4%",
                     cbar_pad=0.15,
                     aspect=False, # Needed if grid not square
                     direction="row")
    
    # Create the data_expanded list that is looped over along the axes when creating the plots
    data_expanded = []
    if parameter == "f":
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx])):
                data_expanded.append(data[dataset_idx][b_set_idx].f_rmse[:,:,:,SNR_idx])
    elif parameter == "D*":
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx])):
                data_expanded.append(data[dataset_idx][b_set_idx].D_star_rmse[:,:,:,SNR_idx])
    elif parameter == "D":
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx])):
                data_expanded.append(data[dataset_idx][b_set_idx].D_rmse[:,:,:,SNR_idx])
    
    # Initiate colormap normalisation
    vmin = 0.99
    vmax = 0
    # Set to global_colorbar_norm to True if the vmin and vmax should be set to the max and min of all D_idx's
    if parameter == "D*":
        exclude = 3
    else:
        exclude = 0
    if global_colorbar_norm:
        for ax_idx in range(len(data_expanded)-exclude):
            if np.max(data_expanded[ax_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx])
            if np.min(data_expanded[ax_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx])
    else:
        for ax_idx in range(len(data_expanded)-exclude):
            if np.max(data_expanded[ax_idx][:,:,D_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx][:,:,D_idx])
            if np.min(data_expanded[ax_idx][:,:,D_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx][:,:,D_idx])
    
    if normscale == "linear":            
        norm=colors.Normalize(vmin=vmin, vmax=vmax)
    elif normscale == "log":
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm=colors.Normalize(vmin=vmin, vmax=vmax)
               
    
    # Lists of indexes of the axes where labels shall be places
    title_labels = list(range(len(data[0])))
    y_labels = [len(data[0])*i for i in range(len(data))]
    print(y_labels)
    x_labels = list(range(len(data_expanded)-len(data[0]), len(data_expanded))) 
    titles = ["Standard", "+ low", "+ intermediate", "+ high"]*len(data)
    ylabels = ["Linear", "sIVIM", "Segmented", "TRR", "VarPro", "TopoPro"]
    if parameter == "D*":
        ylabels = ylabels[2:]
    
    # Fill the image grid
    for ax_idx in range(len(grid)):
        im = grid[ax_idx].pcolormesh(parameter_table["f"], parameter_table["D*"], data_expanded[ax_idx][:,:,D_idx], cmap="gray", norm=norm)
        
            
        if ax_idx in title_labels:
            grid[ax_idx].set_title(f"{titles[ax_idx]}")
        if ax_idx in y_labels:
            grid[ax_idx].set_ylabel(f"{ylabels[int(np.ceil((ax_idx+1)/len(data[0])-1))]} $D^*$")
        if ax_idx in x_labels:
            grid[ax_idx].set_xlabel("$f$", fontsize="large")
    
    m0=vmin           # colorbar min value
    m9=vmax             # colorbar max value
    
    # Colorbar ticks 
    ticks_lower = list(np.linspace(vmin, 0, 4))
    ticks_upper = list(np.linspace(0, vmax, 4))
    ticks = ticks_lower + ticks_upper
    ticks = [tick if round(tick*1000, 1) != 0 else 0 for tick in ticks]
    
    if parameter == "f":
        unit = "[%]"
    else:
        unit = "[µm$^2$/ms]"
    cblabel = f"{parameter} change in bias {unit} for D = {parameter_table['D'][D_idx]*1000:.1f} µm$^2$/ms"
    
    grid[-1].cax.cla()
    cbar = matplotlib.colorbar.Colorbar(grid[-1].cax , im, label=cblabel, ticks=ticks)
    # Set colorbar ticklabels
    if parameter != "f":
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([round(val*1000, 2) for val in ticks])
    else:
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([round(val*100, 2) for val in ticks])
        
    fig.tight_layout()

def plot_sum_rmse_vs_param(data, parameter, SNR_idx=0, parameter_table=parameter_table, b_sets=None, save_name=False):
    plt.style.use(["science", "ieee"]) # ieee uses Times new roman
    plt.rcParams['text.usetex'] = True # Prevents font change when italicized
    #plt.rcParams["font.family"] = "Times New Roman"
    #plt.rcParams.update({
        #"font.family": "Times New Roman"})#,   # specify font family here
        #"font.serif": ["Times"],  # specify font here
        #"font.size":11})          # specify font size here
    
    if parameter == "D*":
        data = data[2:]
    
    fontsize=14
    if parameter == "D*":
        label_parameter = "D^*"
    else:
        label_parameter = parameter
        
    param_lst = ["f", "D*", "D"]
    if not b_sets: 
        b_sets = ["Standard", "+ low", "+ intermediate", "+ high", "Optimized"]
    ylabels = ["Linear", "sIVIM", "Subtracted", "Segmented", "MIX", "TopoPro"]
    linestyles = ["solid", "dashed", "dashdot", (5, (10,3)), (0, (3, 5, 1, 5, 1, 5)), "dotted"]
    linestyles = ["solid", "dashed", "dashdot", (5, (10,3)), (0, (3, 5, 1, 5, 1, 5))]
    if len(data[0]) == 2:
        linestyles = [linestyles[0], linestyles[3]]
    #color_lst = ["black", "red", "green", "blue", "orange", "pink"]
    #color_lst = ["b", "g", "r", "c", "m", "y", "black"]
    color_lst = sns.color_palette('Paired')
    if parameter == "D*":
        ylabels = ylabels[2:]
        color_lst = color_lst[2:]
    
    a4_width_minus_margins = 9.69 # Actually length but makes the lines thinner and the plots prettier
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(a4_width_minus_margins,a4_width_minus_margins/2.5), sharey=True)
    for method_idx in range(len(data)):
        for b_set_idx in range(len(data[0])): 
            if parameter == "f":
                axs[0].plot(parameter_table["f"]*100, np.sum(data[method_idx][b_set_idx].f_rmse[:,:,:,SNR_idx], axis=(0,2))/np.max(np.sum(data[0][1].f_rmse[:,:,:,SNR_idx], axis=(0,2))), label=b_sets[b_set_idx], ls=linestyles[b_set_idx], color=color_lst[method_idx])
                axs[0].set_xlabel("$f$ [\%]", fontsize=fontsize)
                axs[0].tick_params(labelsize=fontsize)
                axs[0].set_xticks([10, 30, 50])
                
                axs[1].plot(parameter_table["D*"]*1000, np.sum(data[method_idx][b_set_idx].f_rmse[:,:,:,SNR_idx], axis=(1,2))/np.max(np.sum(data[0][1].f_rmse[:,:,:,SNR_idx], axis=(1,2))), label=b_sets[b_set_idx], ls=linestyles[b_set_idx], color=color_lst[method_idx])
                axs[1].set_xlabel("$D^*$ [µm$^2$/ms]", fontsize=fontsize)
                axs[1].tick_params(labelsize=fontsize)
                
                axs[2].plot(parameter_table["D"]*1000, np.sum(data[method_idx][b_set_idx].f_rmse[:,:,:,SNR_idx], axis=(0,1))/np.max(np.sum(data[0][1].f_rmse[:,:,:,SNR_idx], axis=(0,1))), label=b_sets[b_set_idx], ls=linestyles[b_set_idx], color=color_lst[method_idx])
                axs[2].set_xlabel("$D$ [µm$^2$/ms]", fontsize=fontsize)
                axs[2].tick_params(labelsize=fontsize)
                #axs[2].legend(loc=0)
            if parameter == "D*":
                axs[0].plot(parameter_table["f"]*100, np.sum(data[method_idx][b_set_idx].D_star_rmse[:,:,:,SNR_idx], axis=(1,2))/np.max(np.sum(data[-1][0].D_star_rmse[:,:,:,SNR_idx], axis=(1,2))), label=b_sets[b_set_idx], ls=linestyles[b_set_idx], color=color_lst[method_idx])
                axs[0].set_xlabel("$f$ [\%]", fontsize=fontsize)
                axs[0].tick_params(labelsize=fontsize)
                axs[0].set_xticks([10, 30, 50])
                
                axs[1].plot(parameter_table["D*"]*1000, np.sum(data[method_idx][b_set_idx].D_star_rmse[:,:,:,SNR_idx], axis=(0,2))/np.max(np.sum(data[-1][0].D_star_rmse[:,:,:,SNR_idx], axis=(0,2))), label=b_sets[b_set_idx], ls=linestyles[b_set_idx], color=color_lst[method_idx])
                axs[1].set_xlabel("$D^*$ [µm$^2$/ms]", fontsize=fontsize)
                axs[1].tick_params(labelsize=fontsize)
                
                axs[2].plot(parameter_table["D"]*1000, np.sum(data[method_idx][b_set_idx].D_star_rmse[:,:,:,SNR_idx], axis=(0,1))/np.max(np.sum(data[-1][0].D_star_rmse[:,:,:,SNR_idx], axis=(0,1))), label=b_sets[b_set_idx], ls=linestyles[b_set_idx], color=color_lst[method_idx])
                axs[2].set_xlabel("$D$ [µm$^2$/ms]", fontsize=fontsize)
                axs[2].tick_params(labelsize=fontsize)
                #axs[2].legend(loc=0)
            if parameter == "D":
                axs[0].plot(parameter_table["f"]*100, np.sum(data[method_idx][b_set_idx].D_rmse[:,:,:,SNR_idx], axis=(1,2))/np.max(np.sum(data[-1][1].D_rmse[:,:,:,SNR_idx], axis=(1,2))), label=b_sets[b_set_idx], ls=linestyles[b_set_idx], color=color_lst[method_idx])
                axs[0].set_xlabel("$f$ [\%]", fontsize=fontsize)
                axs[0].tick_params(labelsize=fontsize)
                axs[0].set_xticks([10, 30, 50])
                
                axs[1].plot(parameter_table["D*"]*1000, np.sum(data[method_idx][b_set_idx].D_rmse[:,:,:,SNR_idx], axis=(0,2))/np.max(np.sum(data[0][1].D_rmse[:,:,:,SNR_idx], axis=(0,2))), label=b_sets[b_set_idx], ls=linestyles[b_set_idx], color=color_lst[method_idx])
                axs[1].set_xlabel("$D^*$ [µm$^2$/ms]", fontsize=fontsize)
                axs[1].tick_params(labelsize=fontsize)
                
                axs[2].plot(parameter_table["D"]*1000, np.sum(data[method_idx][b_set_idx].D_rmse[:,:,:,SNR_idx], axis=(0,1))/np.max(np.sum(data[-1][1].D_rmse[:,:,:,SNR_idx], axis=(0,1))), label=b_sets[b_set_idx], ls=linestyles[b_set_idx], color=color_lst[method_idx])
                axs[2].set_xlabel("$D$ [µm$^2$/ms]", fontsize=fontsize)
                axs[2].tick_params(labelsize=fontsize)
                #axs[2].legend(loc=0)
        
        axs[0].set_ylabel(f"Rel. sum of ${label_parameter}$ RMSE", fontsize=fontsize)
        axs[0].set_xlim(2, 50)
        axs[1].set_xlim(5, 30)
        axs[2].set_xlim(0.5, 3)
    
    #axs[0].set_title("vs $f$")
    #axs[1].set_title("vs $D^*$")
    #axs[2].set_title("vs $D$")
    
    legend_elements = [Line2D([0], [0], color=color_lst[method_idx], lw=1, ls="solid", label=ylabels[method_idx]) for method_idx in range(len(ylabels))]
    fig.legend(handles=legend_elements, loc="lower center", ncol=len(ylabels), frameon=False, bbox_to_anchor=(0.5, -0.12), fontsize=fontsize)
    legend_elements = [Line2D([0], [0], color="black", lw=1, ls=linestyles[b_set_idx], label=b_sets[b_set_idx]) for b_set_idx in range(len(data[0]))]
    fig.legend(handles=legend_elements, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=fontsize)
    fig.suptitle(f"RMSE trends of ${label_parameter}$ at SNR = {int(SNR_table['SNR 0'][SNR_idx])}", fontsize=fontsize)
    
    fig.tight_layout()
    if save_name:
        fig.savefig(f"{save_name}.pdf")

def plot_bias_std_quotient(data, parameter, bvals=bvals, D_idx=3, SNR_idx=0, parameter_table=parameter_table, normscale="linear", global_colorbar_norm=False):
    """
    Plots a bias map for different fitting algorithms and different b-value sets.

    Args:
        data (list): list of lists. First level is fitting algorithms; 2nd level is a list of FitMethod objects for different b-value simulations.
        parameter (string): The parameter to plot the heat maps for. Set to "f", or "D*", or "D".
        bvals (list, optional): List of list of b-value sets. Defaults to bvals.
        D_idx (int, optional): The D_idx to show the heat maps for. Defaults to 3.
        parameter_table (DataFrame, optional): DataFrame object of read parameter_table from simulations. Defaults to parameter_table.
        normscale (str, optional): Change to log-scaled colormap. May not work due to zero-valued elements. Defaults to "linear".
        global_colorbar_norm (bool, optional): Whether to normalsie the colour map to the values of the D_idx-slice, or the entire volume. Defaults to False.
    """
    if parameter == "D*":
        data = data[2:] # Remove the linear  and sIVIM fits, since it does not contain any D* data
    
    # Set up fig and img grid
    fig = plt.figure(figsize=(8,3))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(len(data), len(data[0])),
                     axes_pad=0.1,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="4%",
                     cbar_pad=0.15,
                     aspect=False, # Needed if grid not square
                     direction="row")
    
    # Create the data_expanded list that is looped over along the axes when creating the plots
    data_expanded = []
    if parameter == "f":
        for method_idx in range(len(data)):
            for opt_idx in range(len(data[method_idx])):
                #data_expanded.append(abs(data[method_idx][opt_idx].f_bias)/data[method_idx][opt_idx].f_std[:,:,:,SNR_idx])
                data_expanded.append(data[method_idx][opt_idx].f_rmse[:,:,:,SNR_idx])
    elif parameter == "D*":
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx][1:])):
                data_expanded.append(data[dataset_idx][b_set_idx+1].D_star_bias - data[dataset_idx][0].D_star_bias)
    elif parameter == "D":
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx][1:])):
                data_expanded.append(data[dataset_idx][b_set_idx+1].D_bias - data[dataset_idx][0].D_bias)
    
    # Initiate colormap normalisation
    vmin = 0.99
    vmax = 0
    # Set to global_colorbar_norm to True if the vmin and vmax should be set to the max and min of all D_idx's
    if parameter == "D*":
        exclude = 6
    else:
        exclude = 6
    if global_colorbar_norm:
        for ax_idx in range(len(data_expanded)-exclude):
            if np.max(data_expanded[ax_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx])
            if np.min(data_expanded[ax_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx])
    else:
        for ax_idx in range(len(data_expanded)-exclude):
            if np.max(data_expanded[ax_idx][:,:,D_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx][:,:,D_idx])
            if np.min(data_expanded[ax_idx][:,:,D_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx][:,:,D_idx])
    
    # LogNorm is not an appropriate scale for diverging maps, should be removed or changed
    if normscale == "linear":            
        norm=colors.Normalize(vmin=0, vmax=0.5)
    elif normscale == "log":
        norm=colors.LogNorm(vmin=1, vmax=0)
    else:
        norm=colors.Normalize(vmin=1, vmax=2)
               
    
    # Lists of indexes of the axes where labels shall be places
    title_labels = list(range(len(data[0])))
    y_labels = [3*i for i in range(len(data))]
    x_labels = list(range(len(data_expanded)-len(data[0]), len(data_expanded))) 
    titles = ["Opt no covar", "Opt covar", "Generic"]*len(data)
    ylabels = ["Linear", "sIVIM", "Segmented", "TRR", "VarPro", "TopoPro"]
    if parameter == "D*":
        ylabels = ylabels[2:]
    
    # Fill the image grid
    for ax_idx in range(len(grid)):
        im = grid[ax_idx].pcolormesh(parameter_table["f"], parameter_table["D*"], data_expanded[ax_idx][:,:,D_idx], cmap="gray", norm=norm)
        
            
        if ax_idx in title_labels:
            grid[ax_idx].set_title(f"{titles[ax_idx]}")
        if ax_idx in y_labels:
            grid[ax_idx].set_ylabel(f"{ylabels[int(np.ceil((ax_idx+1)/len(data[0][1:])-1))]} $D^*$")
        if ax_idx in x_labels:
            grid[ax_idx].set_xlabel("$f$", fontsize="large")
    
    m0=vmin           # colorbar min value
    m9=vmax             # colorbar max value
    
    # Colorbar ticks 
    ticks_lower = list(np.linspace(vmin, 0, 4))
    ticks_upper = list(np.linspace(0, vmax, 4))
    ticks = ticks_lower + ticks_upper
    ticks = [tick if round(tick*1000, 1) != 0 else 0 for tick in ticks]
    
    if parameter == "f":
        unit = "[%]"
    else:
        unit = "[µm$^2$/ms]"
    cblabel = f"{parameter} change in bias {unit} for D = {parameter_table['D'][D_idx]*1000:.1f} µm$^2$/ms"
    
    grid[-1].cax.cla()
    cbar = matplotlib.colorbar.Colorbar(grid[-1].cax , im, label=cblabel, ticks=ticks)
    # Set colorbar ticklabels
    if parameter != "f":
        cbar.ax.set_yticklabels([round(val*1000, 2) for val in ticks])
    else:
        cbar.ax.set_yticklabels([round(val*100, 2) for val in ticks])
        
    fig.tight_layout()
    
def plot_topopro_bias(data, D_idx=2, parameter="f", parameter_table=parameter_table, cmap="inferno"):
    #fig = plt.figure(figsize=(8,3))
    #grid = ImageGrid(fig, 111,
                     #nrows_ncols=(1, 3),
                     #axes_pad=0.1,
                     #cbar_location="right",
                     #cbar_mode="single",
                     #cbar_size="4%",
                     #cbar_pad=0.15,
                     #aspect=False, # Needed if grid not square
                     #direction="row")
    
    #grid[0].pcolormesh(parameter_table["f"], parameter_table["D*"], data.f_bias[:,:,D_idx], cmap=cmap)
    #grid[0].set_title("Bias in $f$")
    #grid[1].pcolormesh(parameter_table["f"], parameter_table["D*"], data.D_star_bias[:,:,D_idx], cmap=cmap)
    #grid[1].set_title("Bias in $D^*$")
    #grid[2].pcolormesh(parameter_table["f"], parameter_table["D*"], data.D_bias[:,:,D_idx], cmap=cmap)
    #grid[2].set_title("Bias in $D$")
    #grid[0].set_xlabel("$f$")
    #grid[1].set_xlabel("$f$")
    #grid[2].set_xlabel("$f$")
    #grid[0].set_ylabel("$D^*$")
    
    fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(9,3))
    im1 = axs[0].pcolormesh(parameter_table["f"], parameter_table["D*"], data.f_bias[:,:,D_idx], cmap=cmap)
    im2 = axs[1].pcolormesh(parameter_table["f"], parameter_table["D*"], data.D_star_bias[:,:,D_idx], cmap=cmap)
    im3 = axs[2].pcolormesh(parameter_table["f"], parameter_table["D*"], data.D_bias[:,:,D_idx], cmap=cmap)
    axs[0].set_title("Bias in $f$")
    axs[1].set_title("Bias in $D^*$")
    axs[2].set_title("Bias in $D$")
    axs[0].set_xlabel("$f$")
    axs[1].set_xlabel("$f$")
    axs[2].set_xlabel("$f$")
    axs[0].set_ylabel("$D^*$")
    
    fig.colorbar(im1, ax=axs[0])
    plt.colorbar(im2, ax=axs[1])
    plt.colorbar(im3, ax=axs[2])
    fig.tight_layout()

def plot_rmse_grid_vs_protocol(data, parameter, SNR_idx=0, D_idx=3, parameter_table=parameter_table, SNR_table=SNR_table, normscale="linear", global_colorbar_norm=False, cmap="inferno", exclude_vp=False, scale=1, b_sets=None):
    plt.style.use("default")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True # Prevents font change when italicized
    if parameter == "D*":
        data = data[2:] # Remove the linear  and sIVIM fits, since it does not contain any D* data
    
    # Set up fig and img grid
    fig = plt.figure(figsize=(8,9))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(len(data), len(data[0])),
                     axes_pad=0.1,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="4%",
                     cbar_pad=0.15,
                     aspect=False, # Needed if grid not square
                     direction="row")
    
    noise_idx_list = list(range(len(SNR_table["SNR 0"]+1)))*len(data) # List of noise idx for every occurence of data
    data_expanded = []
    if parameter == "f":
        #data_expanded = [ele.f_rmse for ele in data[i] for i in range(len(data[0]))] # Creates a sorted list with repeated values of data
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx])):
                data_expanded.append(data[dataset_idx][b_set_idx].f_rmse)# - data[dataset_idx][0].f_bias)
    elif parameter == "D*":
        #data_expanded = [ele.D_star_rmse for ele in data[i] for i in range(len(data[0]))] # Creates a sorted list with repeated values of data
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx])):
                data_expanded.append(data[dataset_idx][b_set_idx].D_star_rmse)# - data[dataset_idx][0].f_bias)
    elif parameter == "D":
        #data_expanded = [ele.D_rmse for ele in data[i] for i in range(len(data[0]))] # Creates a sorted list with repeated values of data
        for dataset_idx in range(len(data)):
            for b_set_idx in range(len(data[dataset_idx])):
                data_expanded.append(data[dataset_idx][b_set_idx].D_rmse)# - data[dataset_idx][0].f_bias)
    
    # Initiate colormap normalisation
    vmin = 0.99
    vmax = 0
    # Set to global_colorbar_norm to True if the vmin and vmax should be set to the max and min of all D_idx's
    if global_colorbar_norm:
        for ax_idx in range(len(data_expanded)):
            if np.max(data_expanded[ax_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx])
            if np.min(data_expanded[ax_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx])
    elif exclude_vp:
        for ax_idx in range(len(data_expanded[:-6])):
            if np.max(data_expanded[ax_idx][:,:,D_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx][:,:,D_idx])
            if np.min(data_expanded[ax_idx][:,:,D_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx][:,:,D_idx])
        vmax *= scale
    else:
        for ax_idx in range(len(data_expanded)):
            if np.max(data_expanded[ax_idx][:,:,D_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx][:,:,D_idx])
            if np.min(data_expanded[ax_idx][:,:,D_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx][:,:,D_idx])
    
    if normscale == "linear":            
        norm=colors.Normalize(vmin=vmin, vmax=vmax)
    elif normscale == "log":
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm=colors.Normalize(vmin=vmin, vmax=vmax)
        
    
    # Lists of indexes of the axes where labels shall be placed
    title_labels = list(range(len(data[0])))
    titles = ["Opt no covar", "Opt covar", "Generic"]
    if b_sets != None:
        titles = b_sets
    y_labels = [len(data[0])*i for i in range(len(data))]
    print(y_labels)
    ylabels = ["Linear", "sIVIM", "Segmented", "TRR", "VarPro", "TopoPro"]
    x_labels = list(range(len(data_expanded)-len(SNR_table["SNR 0"]), len(data_expanded)))
    if parameter == "D*":
        ylabels = ylabels[2:]
    
    # Fill the image grid
    #yscale = [parameter_table["D*"][i]/parameter_table["D"][D_idx] for i in range(len(parameter_table["D*"]))]
    yscale = parameter_table["D*"]*1000
    for ax_idx in range(len(grid)):
        im = grid[ax_idx].pcolormesh(parameter_table["f"]*100, yscale, data_expanded[ax_idx][:,:,D_idx,SNR_idx], cmap=cmap, norm=norm)
        
        """
        if np.max(data_expanded[ax_idx].D_rmse[:,:,D_idx, noise_idx_list[ax_idx]]) > vmax:
            vmax = np.max(data_expanded[ax_idx].D_rmse[:,:,D_idx, noise_idx_list[ax_idx]])
        if np.min(data_expanded[ax_idx].D_rmse[:,:,D_idx, noise_idx_list[ax_idx]]) < vmin:
            vmin = np.min(data_expanded[ax_idx].D_rmse[:,:,D_idx, noise_idx_list[ax_idx]])
        """
            
        if ax_idx in title_labels:
            grid[ax_idx].set_title(titles[ax_idx])
        if ax_idx in y_labels:
            grid[ax_idx].set_ylabel(f"{ylabels[int(np.ceil((ax_idx+1)/len(data[0])-1))]}\n$D^*$", fontsize="large")
        if ax_idx in x_labels:
            grid[ax_idx].set_xlabel("$f$", fontsize="large")
    
    # Colorbar ticks 
    ticks = list(np.linspace(vmin, vmax, 9))
    ticks = [tick if round(tick*1000, 1) != 0 else 0 for tick in ticks]
    
    if parameter == "f":
        unit = "[%]"
    else:
        unit = "[µm$^2$/ms]"
    cblabel = f"{parameter} RMSE {unit} for D = {parameter_table['D'][D_idx]*1000:.1f} µm$^2$/ms"
    
    grid[-1].cax.cla()
    cbar = matplotlib.colorbar.Colorbar(grid[-1].cax,im, label=cblabel)
    # Set colorbar ticklabels
    if parameter != "f":
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([round(val*1000, 2) for val in ticks])
    else:
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([round(val*100, 2) for val in ticks])
        
    fig.tight_layout()

def rmse_plot_3d(data, parameter, b_values, snr_levels):
    plt.style.use("default")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True # Prevents font change when italicized
    # data is a list of FitMethod objects, with the length corresponding to
    # number of b-values. Each FitMethod has a summed RMSE vector 
    # (self.{parameter}_rmse_total) with the length of number of SNR levels
    
    # Define the matrix that holds the plane
    # Shape (no_of_bvals, no_of_SNR_levels)
    rmse_vs_bvals_snr = np.zeros((len(data), len(data[0].f_rmse_total)))
    std_vs_bvals_snr = np.zeros((len(data), len(data[0].f_std_sq_total)))
    bias_vs_bvals_snr = np.zeros((len(data), len(data[0].f_rmse_total)))
    
    # Fill the matrix with the RMSE
    if parameter == "f":
        for b_set_idx in range(len(data)):
            rmse_vs_bvals_snr[b_set_idx, :] = data[b_set_idx].f_rmse_total
            std_vs_bvals_snr[b_set_idx, :] = data[b_set_idx].f_std_sq_total
            bias_vs_bvals_snr[b_set_idx, :] = data[b_set_idx].f_bias_sq_total
    if parameter == "D*":
        for b_set_idx in range(len(data)):
            rmse_vs_bvals_snr[b_set_idx, :] = data[b_set_idx].D_star_rmse_total
            std_vs_bvals_snr[b_set_idx, :] = data[b_set_idx].D_star_std_sq_total
            bias_vs_bvals_snr[b_set_idx, :] = data[b_set_idx].D_star_bias_sq_total
    if parameter == "D":
        for b_set_idx in range(len(data)):
            rmse_vs_bvals_snr[b_set_idx, :] = data[b_set_idx].D_rmse_total
            std_vs_bvals_snr[b_set_idx, :] = data[b_set_idx].D_std_sq_total
            bias_vs_bvals_snr[b_set_idx, :] = data[b_set_idx].D_bias_sq_total
    
    #fig, ax = plt.subplots() 
    #im = ax.pcolormesh(snr_levels, b_values, rmse_vs_bvals_snr)
    #fig.colorbar(im, ax=ax)
    
    
    # Surface plot
    X, Y = np.meshgrid(snr_levels, b_values)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    rmse_surf = ax.plot_surface(X, Y, rmse_vs_bvals_snr, edgecolor="red", color="red", lw=0.5, alpha=0.5, label="RMSE")
    std_surf = ax.plot_surface(X, Y, std_vs_bvals_snr, edgecolor="green", color="green", lw=0.5, alpha=0.3, label="$\sqrt{\mathrm{Standard\ deviation}^2}$")
    bias_surf = ax.plot_surface(X, Y, bias_vs_bvals_snr, edgecolor="blue", color="blue", lw=0.5, alpha=0.3, label="$\sqrt{\mathrm{Bias}^2}$")
    ax.set_xlabel("SNR", fontsize=12)
    ax.set_ylabel("b-value [s/mm$^2$]", fontsize=12)
    ax.set_zlabel("Total RMSE [a.u.]", fontsize=12)
    ax.tick_params(labelsize=12)
    #ax.view_init(0, 20)
    ax.view_init(10, 30)
    ax.set_proj_type("persp")
    
    ax.set(ylim=(100,400), xlim=(3, 21), xlabel="SNR", ylabel="b-value [s/mm$^2$]")
    #if parameter == "D*":
        #ax.set(ylim=(10,100), xlim=(3, 21), xlabel="SNR", ylabel="b-value [s/mm$^2$]")
    
    #ax.contour(X, Y, rmse_vs_bvals_snr, zdir="x", offset=3, cmap="inferno")
    #ax.contour(X, Y, rmse_vs_bvals_snr, zdir="y", offset=100, color="red")
    
    rmse_surf._edgecolors2d = rmse_surf._edgecolor3d
    rmse_surf._facecolors2d = rmse_surf._facecolor3d
    std_surf._edgecolors2d = std_surf._edgecolor3d
    std_surf._facecolors2d = std_surf._facecolor3d
    bias_surf._edgecolors2d = bias_surf._edgecolor3d
    bias_surf._facecolors2d = bias_surf._facecolor3d

    ax.legend(ncol=3, frameon=False, fontsize=12, loc="lower center", bbox_to_anchor=(0.5,-0.05))
    ax.set_title(f"Total error of ${parameter}$", x=0.5, y=0.9, fontsize=12)
    
def rmse_plot_3d_all(data1, data2, b_values_low, b_values_intermediate, snr_levels, save_name=False):
    plt.style.use("default")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True # Prevents font change when italicized
    
    
    rmse_vs_bvals_snr_f = np.zeros((len(data1), len(data1[0].f_rmse_total)))
    std_vs_bvals_snr_f = np.zeros((len(data1), len(data1[0].f_std_sq_total)))
    bias_vs_bvals_snr_f = np.zeros((len(data1), len(data1[0].f_rmse_total)))
    
    rmse_vs_bvals_snr_D = np.zeros((len(data1), len(data1[0].f_rmse_total)))
    std_vs_bvals_snr_D = np.zeros((len(data1), len(data1[0].f_std_sq_total)))
    bias_vs_bvals_snr_D = np.zeros((len(data1), len(data1[0].f_rmse_total)))
    
    rmse_vs_bvals_snr_Dstar = np.zeros((len(data2), len(data2[0].f_rmse_total)))
    std_vs_bvals_snr_Dstar = np.zeros((len(data2), len(data2[0].f_std_sq_total)))
    bias_vs_bvals_snr_Dstar = np.zeros((len(data2), len(data2[0].f_rmse_total)))
    
    
    for b_set_idx in range(len(data1)):
        rmse_vs_bvals_snr_f[b_set_idx, :] = data1[b_set_idx].f_rmse_total
        std_vs_bvals_snr_f[b_set_idx, :] = data1[b_set_idx].f_std_sq_total
        bias_vs_bvals_snr_f[b_set_idx, :] = data1[b_set_idx].f_bias_sq_total
    for b_set_idx in range(len(data1)):
        rmse_vs_bvals_snr_D[b_set_idx, :] = data1[b_set_idx].D_rmse_total
        std_vs_bvals_snr_D[b_set_idx, :] = data1[b_set_idx].D_std_sq_total
        bias_vs_bvals_snr_D[b_set_idx, :] = data1[b_set_idx].D_bias_sq_total
    for b_set_idx in range(len(data2)):
        rmse_vs_bvals_snr_Dstar[b_set_idx, :] = data2[b_set_idx].D_star_rmse_total
        std_vs_bvals_snr_Dstar[b_set_idx, :] = data2[b_set_idx].D_star_std_sq_total
        bias_vs_bvals_snr_Dstar[b_set_idx, :] = data2[b_set_idx].D_star_bias_sq_total
    
    fig = plt.figure(figsize=plt.figaspect(1/3)*0.6) 
    X, Y = np.meshgrid(snr_levels, b_values_intermediate)
    
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    rmse_surf_f = ax.plot_surface(X, Y, rmse_vs_bvals_snr_f, edgecolor="C3", color="C3", lw=0.5, alpha=0.5, label="RMSE")
    std_surf_f = ax.plot_surface(X, Y, std_vs_bvals_snr_f, edgecolor="C2", color="C2", lw=0.5, alpha=0.3, label="$\sqrt{\mathrm{Standard\ deviation}^2}$")
    bias_surf_f = ax.plot_surface(X, Y, bias_vs_bvals_snr_f, edgecolor="C0", color="C0", lw=0.5, alpha=0.3, label="$\sqrt{\mathrm{Bias}^2}$")
    ax.set_xlabel("SNR", fontsize=12)
    ax.set_ylabel("b-value [s/mm$^2$]", fontsize=12)
    ax.set_zlabel("Total RMSE [a.u.]", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.view_init(10, 30)
    ax.set_proj_type("persp")
    ax.set_title(f"Error in $f$ vs. intermediate b-value", fontsize=12, x=0.5, y=1)
    ax.set(ylim=(100,400), xlim=(3, 21))
    
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    rmse_surf_D = ax.plot_surface(X, Y, rmse_vs_bvals_snr_D, edgecolor="C3", color="C3", lw=0.5, alpha=0.5)#, label="RMSE")
    std_surf_D = ax.plot_surface(X, Y, std_vs_bvals_snr_D, edgecolor="C2", color="C2", lw=0.5, alpha=0.3)#, label="$\sqrt{\mathrm{Standard\ deviation}^2}$")
    bias_surf_D = ax.plot_surface(X, Y, bias_vs_bvals_snr_D, edgecolor="C0", color="C0", lw=0.5, alpha=0.3)#, label="$\sqrt{\mathrm{Bias}^2}$")
    ax.set_xlabel("SNR", fontsize=12)
    ax.set_ylabel("b-value [s/mm$^2$]", fontsize=12)
    #ax.set_zlabel("Total RMSE [a.u.]", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.view_init(10, 30)
    ax.set_proj_type("persp")
    ax.set_title(f"Error in $D$ vs. intermediate b-value", fontsize=12, x=0.5, y=1)
    ax.set(ylim=(100,400), xlim=(3, 21))
    
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    X, Y = np.meshgrid(snr_levels, b_values_low)
    rmse_surf_Dstar = ax.plot_surface(X, Y, rmse_vs_bvals_snr_Dstar, edgecolor="C3", color="C3", lw=0.5, alpha=0.5)#, label="RMSE")
    std_surf_Dstar = ax.plot_surface(X, Y, std_vs_bvals_snr_Dstar, edgecolor="C2", color="C2", lw=0.5, alpha=0.3)#, label="$\sqrt{\mathrm{Standard\ deviation}^2}$")
    bias_surf_Dstar = ax.plot_surface(X, Y, bias_vs_bvals_snr_Dstar, edgecolor="C0", color="C0", lw=0.5, alpha=0.3)#, label="$\sqrt{\mathrm{Bias}^2}$")
    ax.set_xlabel("SNR", fontsize=12)
    ax.set_ylabel("b-value [s/mm$^2$]", fontsize=12)
    #ax.set_zlabel("Total RMSE [a.u.]", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.view_init(10, 30)
    ax.set_proj_type("persp")
    ax.set_title(f"Error in $D^*$ vs. low b-value", fontsize=12, x=0.5, y=1)
    ax.set(ylim=(10,100), xlim=(3, 21))

    # Legend stuff for 3d plot
    rmse_surf_f._edgecolors2d = rmse_surf_f._edgecolor3d
    rmse_surf_f._facecolors2d = rmse_surf_f._facecolor3d
    std_surf_f._edgecolors2d = std_surf_f._edgecolor3d
    std_surf_f._facecolors2d = std_surf_f._facecolor3d
    bias_surf_f._edgecolors2d = bias_surf_f._edgecolor3d
    bias_surf_f._facecolors2d = bias_surf_f._facecolor3d
    
    fig.legend(ncol=3, frameon=False, fontsize=12, loc="lower center", bbox_to_anchor=(0.5,-0.13))
    fig.tight_layout()
    if save_name:
        fig.savefig(f"{save_name}.pdf", transparent=True, bbox_inches="tight")

def rmse_plot_3d_all_esmrmb(data1, data2, b_values_low, b_values_intermediate, snr_levels, save_name=False):
    plt.style.use("default")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True # Prevents font change when italicized
    
    
    rmse_vs_bvals_snr_f = np.zeros((len(data1), len(data1[0].f_rmse_total)))
    std_vs_bvals_snr_f = np.zeros((len(data1), len(data1[0].f_std_sq_total)))
    bias_vs_bvals_snr_f = np.zeros((len(data1), len(data1[0].f_rmse_total)))
    
    rmse_vs_bvals_snr_D = np.zeros((len(data1), len(data1[0].f_rmse_total)))
    std_vs_bvals_snr_D = np.zeros((len(data1), len(data1[0].f_std_sq_total)))
    bias_vs_bvals_snr_D = np.zeros((len(data1), len(data1[0].f_rmse_total)))
    
    rmse_vs_bvals_snr_Dstar = np.zeros((len(data2), len(data2[0].f_rmse_total)))
    std_vs_bvals_snr_Dstar = np.zeros((len(data2), len(data2[0].f_std_sq_total)))
    bias_vs_bvals_snr_Dstar = np.zeros((len(data2), len(data2[0].f_rmse_total)))
    
    
    for b_set_idx in range(len(data1)):
        rmse_vs_bvals_snr_f[b_set_idx, :] = data1[b_set_idx].f_rmse_total
        std_vs_bvals_snr_f[b_set_idx, :] = data1[b_set_idx].f_std_sq_total
        bias_vs_bvals_snr_f[b_set_idx, :] = data1[b_set_idx].f_bias_sq_total
    for b_set_idx in range(len(data1)):
        rmse_vs_bvals_snr_D[b_set_idx, :] = data1[b_set_idx].D_rmse_total
        std_vs_bvals_snr_D[b_set_idx, :] = data1[b_set_idx].D_std_sq_total
        bias_vs_bvals_snr_D[b_set_idx, :] = data1[b_set_idx].D_bias_sq_total
    for b_set_idx in range(len(data2)):
        rmse_vs_bvals_snr_Dstar[b_set_idx, :] = data2[b_set_idx].D_star_rmse_total
        std_vs_bvals_snr_Dstar[b_set_idx, :] = data2[b_set_idx].D_star_std_sq_total
        bias_vs_bvals_snr_Dstar[b_set_idx, :] = data2[b_set_idx].D_star_bias_sq_total
    
    fig = plt.figure(figsize=plt.figaspect(1/3)*0.6) 
    X, Y = np.meshgrid(snr_levels, b_values_intermediate)
    
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    rmse_surf_f = ax.plot_surface(X, Y, rmse_vs_bvals_snr_f, edgecolor="C3", color="C3", lw=0.5, alpha=0.5, label="RMSE")
    std_surf_f = ax.plot_surface(X, Y, std_vs_bvals_snr_f, edgecolor="C2", color="C2", lw=0.5, alpha=0.3, label="$\sqrt{\mathrm{Standard\ deviation}^2}$")
    bias_surf_f = ax.plot_surface(X, Y, bias_vs_bvals_snr_f, edgecolor="C0", color="C0", lw=0.5, alpha=0.3, label="$\sqrt{\mathrm{Bias}^2}$")
    ax.set_xlabel("SNR", fontsize=12)
    ax.set_ylabel("b-threshold [s/mm$^2$]", fontsize=12)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("Total RMSE [a.u.]", fontsize=12, rotation=90)
    ax.tick_params(labelsize=12)
    ax.view_init(10, 30)
    ax.set_proj_type("persp")
    ax.set_title(f"Error of $f$", fontsize=14, x=0.5, y=1)
    ax.set(ylim=(100,400), xlim=(3, 21))
    
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    X, Y = np.meshgrid(snr_levels, b_values_low)
    rmse_surf_Dstar = ax.plot_surface(X, Y, rmse_vs_bvals_snr_Dstar, edgecolor="C3", color="C3", lw=0.5, alpha=0.5)#, label="RMSE")
    std_surf_Dstar = ax.plot_surface(X, Y, std_vs_bvals_snr_Dstar, edgecolor="C2", color="C2", lw=0.5, alpha=0.3)#, label="$\sqrt{\mathrm{Standard\ deviation}^2}$")
    bias_surf_Dstar = ax.plot_surface(X, Y, bias_vs_bvals_snr_Dstar, edgecolor="C0", color="C0", lw=0.5, alpha=0.3)#, label="$\sqrt{\mathrm{Bias}^2}$")
    ax.set_xlabel("SNR", fontsize=12)
    ax.set_ylabel("b-threshold [s/mm$^2$]", fontsize=12)
    #ax.set_zlabel("Total RMSE [a.u.]", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.view_init(10, 30)
    ax.set_proj_type("persp")
    ax.set_title(f"Error of $D^*$", fontsize=14, x=0.5, y=1)
    ax.set(ylim=(100,400), xlim=(3, 21))
    
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    rmse_surf_D = ax.plot_surface(X, Y, rmse_vs_bvals_snr_D, edgecolor="C3", color="C3", lw=0.5, alpha=0.5)#, label="RMSE")
    std_surf_D = ax.plot_surface(X, Y, std_vs_bvals_snr_D, edgecolor="C2", color="C2", lw=0.5, alpha=0.3)#, label="$\sqrt{\mathrm{Standard\ deviation}^2}$")
    bias_surf_D = ax.plot_surface(X, Y, bias_vs_bvals_snr_D, edgecolor="C0", color="C0", lw=0.5, alpha=0.3)#, label="$\sqrt{\mathrm{Bias}^2}$")
    ax.set_xlabel("SNR", fontsize=12)
    ax.set_ylabel("b-threshold [s/mm$^2$]", fontsize=12)
    #ax.set_zlabel("Total RMSE [a.u.]", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.view_init(10, 30)
    ax.set_proj_type("persp")
    ax.set_title(f"Error of $D$", fontsize=14, x=0.5, y=1)
    ax.set(ylim=(100,400), xlim=(3, 21))
    

    # Legend stuff for 3d plot
    rmse_surf_f._edgecolors2d = rmse_surf_f._edgecolor3d
    rmse_surf_f._facecolors2d = rmse_surf_f._facecolor3d
    std_surf_f._edgecolors2d = std_surf_f._edgecolor3d
    std_surf_f._facecolors2d = std_surf_f._facecolor3d
    bias_surf_f._edgecolors2d = bias_surf_f._edgecolor3d
    bias_surf_f._facecolors2d = bias_surf_f._facecolor3d
    
    fig.legend(ncol=3, frameon=False, fontsize=12, loc="lower center", bbox_to_anchor=(0.5,-0.13))
    fig.tight_layout()
    if save_name:
        fig.savefig(f"{save_name}.svg", transparent=True, bbox_inches="tight")

def plot_rmse_grid_esmrmb(data, parameter, D_idx=3, parameter_table=parameter_table, SNR_table=SNR_table, normscale="linear", global_colorbar_norm=False, cmap="inferno", fontsize=12, save_name=None, ylabels=None, title=None, figsize=None, cbar_size=.3, crop_Dstar=True, SNR_idx=0):
    plt.style.use("default")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True # Prevents font change when italicized
    
    if figsize:
        figsize = figsize
    else:
        figsize = (6.27, 9.69)
    if parameter == "D*" and crop_Dstar:
        data = data[2:] # Remove the linear  and sIVIM fits, since it does not contain any D* data
        
        if figsize:
            figsize = figsize
        else:
            figsize = (6.27,6.46)
        
    
    # Set up fig and img grid
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(len(data), 2),
                     axes_pad=0.1,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size=cbar_size,
                     cbar_pad=0.15,
                     aspect=False, # Needed if grid not square
                     direction="row")
    
    data_expanded = []
    if parameter == "f":
        data_expanded = [ele.f_rmse for ele in data for i in range(2)] # Creates a sorted list with repeated values of data
    elif parameter == "D*":
        data_expanded = [ele.D_star_rmse for ele in data for i in range(2)]
    elif parameter == "D":
        data_expanded = [ele.D_rmse for ele in data for i in range(2)]
    
    # Initiate colormap normalisation
    vmin = 0.99
    vmax = 0
    # Set to global_colorbar_norm to True if the vmin and vmax should be set to the max and min of all D_idx's
    if global_colorbar_norm:
        for ax_idx in range(len(data_expanded)):
            if np.max(data_expanded[ax_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx])
            if np.min(data_expanded[ax_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx])
    else:
        for ax_idx in range(len(data_expanded)):
            if np.max(data_expanded[ax_idx][:,:,D_idx]) > vmax:
                vmax = np.max(data_expanded[ax_idx][:,:,D_idx])
            if np.min(data_expanded[ax_idx][:,:,D_idx]) < vmin:
                vmin = np.min(data_expanded[ax_idx][:,:,D_idx])
    
    if normscale == "linear":
        norm=colors.Normalize(vmin=vmin, vmax=vmax)
    elif normscale == "log":
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm=colors.Normalize(vmin=vmin, vmax=vmax)
        
    
    # Lists of indexes of the axes where labels shall be placed
    y_labels = [i for i in range(len(data))]
    y_labels = [0, 2, 4]
    if ylabels:
        ylabels = ylabels
    else:
        ylabels = ["Linear", "sIVIM", "Subtracted", "Segmented", "MIX", "TopoPro"]
    x_labels = list(range(len(data_expanded)-1, len(data_expanded)))
    x_labels = [4, 5]
    if parameter == "D*" and crop_Dstar:
        ylabels = ylabels[2:]
    
    # Fill the image grid
    #yscale = [parameter_table["D*"][i]/parameter_table["D"][D_idx] for i in range(len(parameter_table["D*"]))]
    yscale = parameter_table["D*"]*1000
    for ax_idx in range(0, len(grid)):
        if ax_idx % 2 == 0:
            im = grid[ax_idx].pcolormesh(parameter_table["f"]*100, yscale, data_expanded[ax_idx][:,:,D_idx,SNR_idx], cmap=cmap, norm=norm)
        else:
            grid[ax_idx].pcolormesh(parameter_table["f"]*100, yscale, data_expanded[ax_idx][:,:,D_idx,SNR_idx+1], cmap=cmap, norm=norm)
        
        """
        if np.max(data_expanded[ax_idx].D_rmse[:,:,D_idx, noise_idx_list[ax_idx]]) > vmax:
            vmax = np.max(data_expanded[ax_idx].D_rmse[:,:,D_idx, noise_idx_list[ax_idx]])
        if np.min(data_expanded[ax_idx].D_rmse[:,:,D_idx, noise_idx_list[ax_idx]]) < vmin:
            vmin = np.min(data_expanded[ax_idx].D_rmse[:,:,D_idx, noise_idx_list[ax_idx]])
        """
            
        grid[0].set_title("SNR = 3", fontsize=fontsize)
        grid[1].set_title("SNR = 10", fontsize=fontsize)
        #if ax_idx in range(len(data)):
        if ax_idx in y_labels:
            grid[ax_idx].set_ylabel(f"{ylabels[int(ax_idx/2)]} \n $D^*$ [µm$^2$/ms]", fontsize=fontsize)
        if ax_idx in x_labels:
            grid[ax_idx].set_xlabel("$f$ [\%]", fontsize=fontsize)
            grid[ax_idx].set_xticks([10, 30, 50])
        
        for ax_idx in range(len(grid)):
            grid[ax_idx].tick_params(labelsize=fontsize)
    
    # Colorbar ticks 
    ticks = list(np.linspace(vmin, vmax, 9))
    #ticks = [tick if round(tick*1000, 1) != 0 else 0 for tick in ticks]
    
    if parameter == "f":
        unit = "[\%]"
    else:
        unit = "[µm$^2$/ms]"
    
    if parameter == "D*":
        label_parameter = "D^*"
    else:
        label_parameter = parameter    
    
    cblabel = f"${label_parameter}$ RMSE {unit}"
    
    grid[-1].cax.cla()
    cbar = matplotlib.colorbar.Colorbar(grid[-1].cax,im, label=cblabel)
    cbar.set_label(cblabel, size=fontsize)
    # Set colorbar ticklabels
    if parameter == "D*":
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([int(round(val*1000, 0)) for val in ticks], fontsize=fontsize)
    elif parameter == "D":
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([np.ceil(val*10000)/10 for val in ticks], fontsize=fontsize)
    elif parameter == "f":
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([int(np.ceil(val*100)) for val in ticks], fontsize=fontsize)
        
    fig.tight_layout()
    if save_name:
        fig.savefig(f"{save_name}.png", dpi=300)