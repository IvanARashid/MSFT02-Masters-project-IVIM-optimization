import numpy as np
import pandas as pd
import os
from scipy.stats import mannwhitneyu, ttest_ind

base_path = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\in-vivo statistik"

class FitMethod:
    def __init__(self, method, base_path):
        self.path = base_path
        self.method = method
        
        self.load_data()
        self.calculate_convergence_ratios()
        
    def load_data(self):
        self.data = pd.read_csv(os.path.join(self.path, f"{self.method} convergence table.csv"))
        
    def calculate_convergence_ratios(self):
        self.f_ratios = self.data["f opt"]/self.data["f gen"]
        if self.method in ["Subtracted", "Segmented", "MIX", "TopoPro"]:
            self.Dstar_ratios = self.data["D* opt"]/self.data["D* gen"]
        
fit_methods = ["Linear", "sIVIM", "Subtracted", "Segmented", "MIX", "TopoPro"]

data = [FitMethod(method, base_path) for method in fit_methods]

f_df = pd.DataFrame({})
Dstar_df = pd.DataFrame({})
for method in data:
    f_df[method.method] = method.f_ratios
for method in data[2:]:
    Dstar_df[method.method] = method.Dstar_ratios
    


# u-test for each method
f_results = pd.DataFrame({})
for fit_method in fit_methods:
    convergence_ratio_generic = np.ones(f_df[fit_method].shape) # The ratio for the generic protocol is 1 by definition
    convergence_ratio_optimized = f_df[fit_method]
    
    statistic, pvalue = mannwhitneyu(convergence_ratio_optimized, convergence_ratio_generic, alternative="greater")
    #statistic, pvalue = ttest_ind(convergence_ratio_optimized, convergence_ratio_generic, alternative="greater")
    f_results[fit_method] = [pvalue]

Dstar_results = pd.DataFrame({})
for fit_method in fit_methods[2:]:
    convergence_ratio_generic = np.ones(Dstar_df[fit_method].shape) # The ratio for the generic protocol is 1 by definition
    convergence_ratio_optimized = Dstar_df[fit_method]
    
    statistic, pvalue = mannwhitneyu(convergence_ratio_optimized, convergence_ratio_generic, alternative="greater")
    #statistic, pvalue = ttest_ind(convergence_ratio_optimized, convergence_ratio_generic, alternative="greater")
    Dstar_results[fit_method] = [pvalue]

# Average
f_avg = pd.DataFrame({})
for fit_method in fit_methods:
    f_avg[fit_method] = [np.average(f_df[fit_method]), np.std(f_df[fit_method])]
    
Dstar_avg = pd.DataFrame({})
for fit_method in fit_methods[2:]:
    Dstar_avg[fit_method] = [np.average(Dstar_df[fit_method]), np.std(Dstar_df[fit_method])]