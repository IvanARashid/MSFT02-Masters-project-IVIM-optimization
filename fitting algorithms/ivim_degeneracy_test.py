import numpy as np
import ivim_fit_degeneracy_test
from dipy.core.gradients import gradient_table
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots

plt.style.use(["science", "ieee"])

def ivim_signal(b, S0, f, D_star, D):
    return S0*(f*np.exp(-b*D_star) + (1-f)*np.exp(-b*D))

def generate_noise(loc, sigma):
    real_component = norm.rvs(loc=loc, scale=sigma)
    imaginary_component = norm.rvs(loc=loc, scale=sigma)
    return complex(real_component, imaginary_component)

def add_rician_noise(signal, SNR):
    sigma = signal[-1]/SNR
    # Sample real and imaginary noise components from gaussian distributions
    # Use the last b-value as the SNR baseline in order to avoid the noise floor
    noise = np.array([generate_noise(signal_value, sigma) for signal_value in signal])
    
    # Add the two components to the signal and take the magniutde of the result
    noised_signal = signal + noise
    noised_signal = np.absolute(noised_signal)
    return noised_signal
    
def normalized_residual_variance(fit_signals, noise_signals, signal, SNR):
    sigma = signal[-1]/SNR # Variance of noise
    NRV_array = np.zeros((len(noise_signals))) # Create NRV array
    
    # Björn Lampinen PhD thesis
    for noise_realization in range(len(noise_signals)):
        NRV = 0 
        for measurement in range(len(noise_signals[noise_realization])):
            NRV += ((noise_signals[noise_realization, measurement] - fit_signals[noise_realization, measurement])**2/(len(noise_signals[noise_realization])-3))/sigma**2
            
        NRV_array[noise_realization] = NRV
    
    return NRV_array

def single_f_evaluation(f, noise_realizations, gtab, S0, f_truth, D_star, D, SNR):
    # Generate un-noised signal
    signal = ivim_signal(gtab.bvals, S0, f_truth, D_star, D)
    
    # Simulate the noise
    noise_signals = np.zeros((noise_realizations, len(signal)))
    for i in range(noise_realizations):
        noise_signals[i,:] = add_rician_noise(signal, SNR)
    
    # Perform fits
    ivim_model = ivim_fit_degeneracy_test.IvimModelBiExp(gtab, f_fix=f)
    ivim_fits = ivim_model.fit(noise_signals)
    
    # Generate signals from parameter estimates
    fit_signals = np.zeros((noise_realizations, len(signal)))
    for i in range(noise_realizations):
        fit_signals[i,:] = ivim_signal(gtab.bvals, *ivim_fits.model_params[i])
    
    # Get the NRV
    NRV_array = normalized_residual_variance(fit_signals, noise_signals, signal, SNR)
    return NRV_array

def evaluate_NRV_for_SNR(f_fix, noise_realizations, gtab, S0, f, D_star, D, SNR):
    NRV_array_for_all_f = np.zeros((len(f_fix), noise_realizations))
    for f_idx in range(len(f_fix)):
        NRV_array_for_all_f[f_idx, :] = single_f_evaluation(f_fix[f_idx], noise_realizations, gtab, S0, f, D_star, D, SNR)
        
    NRV_averaged = np.average(NRV_array_for_all_f, axis=1)
    NRV_data1 = NRV_array_for_all_f[:, 0]
    NRV_data2 = NRV_array_for_all_f[:, 1]
    NRV_data3 = NRV_array_for_all_f[:, 2]
    return NRV_averaged, NRV_data1, NRV_data2, NRV_data3

def plot_NRV_in_subplot(NRV_data, ax, SNR):
    ax.plot(f_fix, NRV_data[0], color="black", ls="-", marker="", lw=.7)
    ax.plot(f_fix, NRV_data[1], ls="-", alpha=0.3, lw=.7)
    ax.plot(f_fix, NRV_data[2], ls="-", alpha=0.3, lw=.7)
    ax.plot(f_fix, NRV_data[3], ls="-", alpha=0.3, lw=.7)
    ax.set_xlabel("$f$")
    ax.axvline(x=f, color="black", ls="--", lw=0.7)
    ax.set_title(f"SNR {SNR}")
    ax.set_ylim(1, np.max(NRV_data[0]))
    


# Define b-values and construct gradient table
bvals = np.array([0,20,40,60,80,100,150,200,300,400,500,600,700,800])
#bvals = np.array([0,0,50,50,50,250,250,250,250,250,800,800,800,800,800])
bvec = np.zeros((bvals.size, 3))
bvec[:,2] = 1
gtab = gradient_table(bvals, bvec, b0_threshold=0)

# Ground truth
S0 = 1
D_star = 30e-3
D = 1e-3
f = 0.1

# Settings
f_fix = np.linspace(0, 1, 10)
noise_realizations = 100
SNR = 30

data_SNR_10 = evaluate_NRV_for_SNR(f_fix, noise_realizations, gtab, S0, f, D_star, D, 10)
data_SNR_20 = evaluate_NRV_for_SNR(f_fix, noise_realizations, gtab, S0, f, D_star, D, 20)
data_SNR_30 = evaluate_NRV_for_SNR(f_fix, noise_realizations, gtab, S0, f, D_star, D, 30)



width = 7
fig, axs = plt.subplots(ncols=3, figsize=(width, width/3))
plot_NRV_in_subplot(data_SNR_10, axs[0], 10)
plot_NRV_in_subplot(data_SNR_20, axs[1], 20)
plot_NRV_in_subplot(data_SNR_30, axs[2], 30)
axs[0].set_ylabel("NRV")
fig.tight_layout()


