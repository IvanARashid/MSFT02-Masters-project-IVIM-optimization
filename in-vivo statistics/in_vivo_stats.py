import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os
from scipy.stats import mannwhitneyu

plt.style.use(["science", "ieee"])

base_path = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\in-vivo statistik"

class FitMethod:
    def __init__(self, method, base_path=base_path):
        self.path = base_path
        self.method = method
        
        self.f_opt, self.f_opt_h, self.f_gen, self.f_gen_h = self.load_data("f")*1e2
        self.D_opt, self.D_opt_h, self.D_gen, self.D_gen_h = self.load_data("D")*1e3
        
        if self.method in ["Subtracted", "Segmented", "MIX", "TopoPro"]:
            self.Dstar_opt, self.Dstar_opt_h, self.Dstar_gen, self.Dstar_gen_h = self.load_data("Dstar")*1e3
        
    def load_data(self, parameter):
        filename_opt = f"{self.method} {parameter} opt.npy"
        filename_opt_h = f"{self.method} {parameter} opt h.npy"
        filename_gen = f"{self.method} {parameter} gen.npy"
        filename_gen_h = f"{self.method} {parameter} gen h.npy"
        
        optimized = np.load(os.path.join(self.path, filename_opt))
        optimized_h = np.load(os.path.join(self.path, filename_opt_h))
        generic = np.load(os.path.join(self.path, filename_gen))
        generic_h = np.load(os.path.join(self.path, filename_gen_h))
        
        return np.array([optimized, optimized_h, generic, generic_h])

fit_methods = ["Linear", "sIVIM", "Subtracted", "Segmented", "MIX", "TopoPro"]
labels = []
for i in range(len(fit_methods)):
    labels.extend([f"{fit_methods[i]} gen.", f"{fit_methods[i]} opt."])

data = [FitMethod(method) for method in fit_methods]
f_data = []
f_data_h = []
Dstar_data = []
Dstar_data_h = []
D_data = []
D_data_h = []

fopt = []
fgen = []

for method in data:
    f_data.extend((method.f_gen, method.f_opt))
    f_data_h.extend((method.f_gen_h, method.f_opt_h))
    fopt.append(method.f_opt)
    fgen.append(method.f_gen)

    D_data.extend((method.D_gen, method.D_opt))
    D_data_h.extend((method.D_gen_h, method.D_opt_h))

    if method.method in ["Subtracted", "Segmented", "MIX", "TopoPro"]:
        Dstar_data.extend((method.Dstar_gen, method.Dstar_opt))
        Dstar_data_h.extend((method.Dstar_gen_h, method.Dstar_opt_h))
        

width = 7

res = mannwhitneyu(f_data[0], f_data_h[0], alternative="less")
print(res)

color_palette = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c']
colors = []
for color in color_palette:
    colors.extend((color, color))

fontsize=10
"""
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(width,width), sharex=True)
#vplot1 = axs[0,0].violinplot(f_data, showmedians=True, showextrema=False, quantiles=[[0.25, 0.75]]*12, bw_method="silverman")
for i in range(len(f_data)):
    y = f_data[i]
    x = np.random.normal(1+i, 0.09, size=len(y))
    axs[0,0].plot(x, y, alpha=0.05, color="gray", marker=".", ls="")
bplot1 = axs[0,0].boxplot(f_data, notch=True, showfliers=False, patch_artist=True)
axs[0,0].set_ylim(-1,40)
axs[0,0].set_ylabel("$f$ [\%]", fontsize=fontsize)
axs[0,0].set_xticks(range(1, len(fit_methods)*2+1))
axs[0,0].set_xticklabels([], rotation=0)
axs[0,0].tick_params(labelsize=fontsize)
axs[0,0].set_title("Tumour ROI", fontsize=fontsize)

#vplot2 = axs[0,1].violinplot(f_data_h, showmedians=True, showextrema=False, quantiles=[[0.25, 0.75]]*12)
#for i in range(len(f_data)):
    #y = f_data_h[i]
    #x = np.random.normal(1+i, 0.09, size=len(y))
    #axs[0,1].plot(x, y, alpha=0.05, color="gray", marker=".", ls="")
bplot2 = axs[0,1].boxplot(f_data_h, notch=True, showfliers=False, patch_artist=True)
axs[0,1].set_ylim(-1,40)
axs[0,1].set_yticklabels([])
axs[0,1].set_xticks(range(1, len(fit_methods)*2+1))
axs[0,1].set_xticklabels([], rotation=0)
axs[0,1].tick_params(labelsize=10)
axs[0,1].set_title("Prostate minus tumour ROI", fontsize=fontsize)

#vplot5 = axs[1,0].violinplot(Dstar_data, showmedians=True, showextrema=False, positions=range(5,len(Dstar_data)+5), quantiles=[[0.25, 0.75]]*8)
for i in range(len(Dstar_data)):
    y = Dstar_data[i]
    x = np.random.normal(5+i, 0.09, size=len(y))
    axs[1,0].plot(x, y, alpha=0.05, color="gray", marker=".", ls="")
bplot5 = axs[1,0].boxplot(Dstar_data, notch=True, showfliers=False, positions=range(5,len(Dstar_data)+5), patch_artist=True)
axs[1,0].set_ylim(0,180)
axs[1,0].set_ylabel("$D^*$ [µm$^2$/ms]", fontsize=fontsize)
axs[1,0].set_xticks(range(1, len(fit_methods)*2+1))
axs[1,0].set_xticklabels(["", "", "", ""] + labels[4:], rotation=90)
axs[1,0].tick_params(labelsize=10)

#vplot6 = axs[1,1].violinplot(Dstar_data_h, showmedians=True, showextrema=False, positions=range(5,len(Dstar_data)+5), quantiles=[[0.25, 0.75]]*8)
bplot6 = axs[1,1].boxplot(Dstar_data_h, notch=True, showfliers=False, positions=range(5,len(Dstar_data)+5), patch_artist=True)
axs[1,1].set_ylim(0,180)
axs[1,1].set_yticklabels([])
axs[1,1].set_xticks(range(1, len(fit_methods)*2+1))
axs[1,1].set_xticklabels(["", "", "", ""] + labels[4:], rotation=90)
axs[1,1].tick_params(labelsize=10)

#vplot3 = axs[2,0].violinplot(D_data, showmedians=True, showextrema=False, quantiles=[[0.25, 0.75]]*12)
for i in range(len(D_data)):
    y = D_data[i]
    x = np.random.normal(1+i, 0.09, size=len(y))
    axs[2,0].plot(x, y, alpha=0.05, color="gray", marker=".", ls="")
bplot3 = axs[2,0].boxplot(D_data, notch=True, showfliers=False, patch_artist=True)
axs[2,0].set_ylim(0.4,2.5)
axs[2,0].set_ylabel("$D$ [µm$^2$/ms]", fontsize=fontsize)
axs[2,0].set_xticks(range(1, len(fit_methods)*2+1))
axs[2,0].set_xticklabels(labels, rotation=90)
axs[2,0].tick_params(labelsize=10)

#vplot4 = axs[2,1].violinplot(D_data_h, showmedians=True, showextrema=False, quantiles=[[0.25, 0.75]]*12)
bplot4 = axs[2,1].boxplot(D_data_h, notch=True, showfliers=False, patch_artist=True)
axs[2,1].set_ylim(0.4,2.5)
axs[2,1].set_yticklabels([])
axs[2,1].set_xticks(range(1, len(fit_methods)*2+1))
axs[2,1].set_xticklabels(labels, rotation=90)
axs[2,1].tick_params(labelsize=10)


"""
"""
for bplot in (vplot1, vplot2, vplot3, vplot4):
    for patch, color in zip(bplot['bodies'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#793636")
        
for bplot in (vplot5, vplot6):
    for patch, color in zip(bplot['bodies'], colors[4:]):
        patch.set_facecolor(color)
        patch.set_edgecolor("#793636")
for bplot in (vplot1, vplot2, vplot3, vplot4, vplot5, vplot6):
    bplot["cmedians"].set_edgecolor('#e31a1c')
    bplot["cquantiles"].set_edgecolor("#793636")
    bplot["cquantiles"].set_alpha(0.2)
"""
"""
        
for bplot in (bplot1, bplot2, bplot3, bplot4):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for patch, color in zip(bplot['whiskers'], colors):
        patch.set_color("black")
        patch.set_alpha(0.5)
        
for bplot in (bplot5, bplot6):
    for patch, color in zip(bplot['boxes'], colors[4:]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for patch, color in zip(bplot['whiskers'], colors[4:]):
        patch.set_color("black")
        patch.set_alpha(0.5)
fig.tight_layout()
"""

fig2, axs = plt.subplots(ncols=3, figsize=(width, width/3))
axs[0].hist(data[3].f_gen_h, bins=50, range=(0, 50), density=True, color=color_palette[3], alpha=1, label="Segmented", histtype="step", ls="solid")
axs[0].hist(data[4].f_gen_h, bins=50, range=(0, 50), density=True, color=color_palette[4], alpha=1, label="MIX", histtype="step", ls="solid")
axs[0].hist(data[5].f_gen_h, bins=50, range=(0, 50), density=True, color=color_palette[5], alpha=1, label="MIX", histtype="step", ls="solid")
axs[0].set_xlabel("$f$ [\%]", fontsize=fontsize)
axs[0].set_xticks([0,10,20,30,40,50])
axs[0].set_ylabel("Probability density", fontsize=fontsize)
axs[0].tick_params(labelsize=fontsize)

axs[1].hist(data[3].Dstar_gen_h, bins=50, range=(5, 100), density=True, color=color_palette[3], alpha=1, histtype="step", ls="solid")
axs[1].hist(data[4].Dstar_gen_h, bins=50, range=(5, 100), density=True, color=color_palette[4], alpha=1, histtype="step", ls="solid")
axs[1].set_xlabel("$D^*$ [µm$^2$/ms]", fontsize=fontsize)
axs[1].set_xticks([0,20,40,60,80,100])
axs[1].tick_params(labelsize=fontsize)

axs[2].hist(data[3].D_gen_h, bins=50, range=(0, 3), density=True, color=color_palette[3], alpha=1, histtype="step", ls="solid")
axs[2].hist(data[4].D_gen_h, bins=50, range=(0, 3), density=True, color=color_palette[4], alpha=1, histtype="step", ls="solid")
axs[2].hist(data[5].D_gen_h, bins=50, range=(0, 3), density=True, color=color_palette[5], alpha=1, histtype="step", ls="solid")
axs[2].set_xlabel("$D$ [µm$^2$/ms]", fontsize=fontsize)
axs[2].tick_params(labelsize=fontsize)
axs[2].set_xlim(0,3)

#axs[0].legend(fontsize=fontsize)
fig2.suptitle("Probability density of parameter estimates by", fontsize=fontsize, x=0.45)
fig2.legend(fontsize=fontsize, bbox_to_anchor=(0.8, 1.07))
fig2.tight_layout()