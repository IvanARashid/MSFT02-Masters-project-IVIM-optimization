import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def ivim_signal(b, f, D_star, D, S0=1):
    return S0*(f*np.exp(-b*D_star) + (1-f)*np.exp(-b*D))

def get_gradient(value_range, n, geom_spaced=False):
    # Function that creates the gradients
    # Has an option to create a geometrically spaced gradient
    if geom_spaced:
        gradient = np.geomspace(*value_range, n)
    else:
        gradient = np.linspace(*value_range, n)
    return gradient

# Creates the gradients
n = 10
f = get_gradient((0.02, 0.5), n)
D_star = get_gradient((5, 30), n)
D = get_gradient((0.05, 3), n)

# Create a table output for the parameter values
parameter_dict = {"f" : f,
                    "D*" : D_star,
                    "D" : D}
parameter_df = pd.DataFrame.from_dict(parameter_dict)

# Create the phantom volume, with each voxel having a unique set of parameter values
f, D_star, D = np.meshgrid(f, D_star, D)

# b-values for the simulated signals
b_values = np.array([0, .05, .240, .800])

signal_volume = np.array([ivim_signal(bval, f, D_star, D) for bval in b_values])
signal_volume = np.transpose(signal_volume, axes=[1,2,3,0])    
    
plt.plot(b_values, signal_volume[5,5,5,:])

#outputs["Simulated signal"] = hero.HeroImage(signal)
#outputs["b-values"] = hero.HeroArray(b_values)
#outputs["f ground truth"] = hero.HeroImage(f)
#outputs["D* ground truth"] = hero.HeroImage(D_star)
#outputs["D ground truth"] = hero.HeroImage(D)

def parameter_array(number_of_elements, f_range, D_star_range, D_range, b_values)