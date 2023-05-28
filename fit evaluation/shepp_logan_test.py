# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:15:25 2023

@author: Ivan
"""

import numpy as np
from tqdm import tqdm
from phantominator import mr_shepp_logan
import matplotlib.pyplot as plt
from shepp_logan import ivim_shepp_logan

def ivim_signal(b, S0, f, D_star, D):
    S = S0*(f*np.exp(-b*D_star) + (1-f)*np.exp(-b*D))
    return S

M0, T1, T2 = mr_shepp_logan((160,160,1), zlims=(-.25,.25))

#plt.imshow(M0, cmap="gray")

S0, f, D_star, D = ivim_shepp_logan((160,160,1), zlims=(-.25,.25))

fig, axs = plt.subplots(nrows=2, ncols=2)
axs[0,0].imshow(S0, cmap="gray")
axs[0,0].set_title("S0")

axs[0,1].imshow(f, cmap="gray")
axs[0,1].set_title("f")

axs[1,0].imshow(D, cmap="gray")
axs[1,0].set_title("D")

axs[1,1].imshow(D_star, cmap="gray")
axs[1,1].set_title("D*")

fig.tight_layout()

b_values = [0, 50, 240, 800]
S = np.zeros((S0.shape[0], S0.shape[1], len(b_values)))
for i in range(len(b_values)):
    for row in tqdm(range(S0.shape[0])):
        for column in range(S0.shape[1]):
            S[row, column, i] = ivim_signal(b_values[i], S0[row, column, 0], f[row, column, 0], D_star[row, column, 0], D[row, column, 0])
    

fig, axs = plt.subplots(ncols=len(b_values))
for i in range(len(b_values)):
    axs[i].imshow(S[:,:,i], cmap="gray")
    axs[i].set_title("b = {}".format(b_values[i]))
fig.tight_layout()
    