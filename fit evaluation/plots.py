import bias_std_rmse_plot as functions
import os

path_report_figures = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Rapport\appendix\figures"
path_esmrmb_figures = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\ESMRMB"

path_initial_final = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\new opts\initial\b[0, 15, 30, 50, 240, 800] SNR[3, 10, 30] samples 100"
path_initial_opt = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\new opts\initial_new\b[0, 50, 240, 800] SNR[3, 10, 30] samples 100"
path_no_covar_opt = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\new opts\opt no covar_new\b[0, 80, 350, 800] SNR[3, 10, 30] samples 100"
path_no_covar_final = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\new opts\opt no covar\b[0, 30, 50, 80, 350, 800] SNR[3, 10, 30] samples 100"
path_generic = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\new opts\generic_new\b[0, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800] SNR[3, 10, 30] samples 100"
path_covar_five = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\new opts\opt covar_new\b[0, 60, 70, 220, 310, 350, 520, 800] SNR[3, 10, 30] samples 100"
path_covar_fifty = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\new opts\opt covar\b[0, 70, 200, 320, 340, 350, 380, 800] SNR[3, 10, 30] samples 100"

base_path_3d = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\new opts\3d_single_NEX"
base_path_3d = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\new opts\3d_single_NEX"
intermediary_bvals = [100, 150, 200, 250, 300, 350, 400]
intermediary_folder_names_3d = [f"b[0, 50, {intermediary_bval}, 800] SNR[3, 6, 9, 12, 15, 18, 21] samples 50" \
                                for intermediary_bval in intermediary_bvals]
sivim_dataset_3d = [functions.FitMethod(os.path.join(base_path_3d, folder_name), "sIVIM") \
                    for folder_name in intermediary_folder_names_3d]

low_bvals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
low_folder_names_3d = [f"b[0, {low_bval}, 250, 800] SNR[3, 6, 9, 12, 15, 18, 21] samples 50" \
                       for low_bval in low_bvals]
trr_dataset_3d = [functions.FitMethod(os.path.join(base_path_3d, folder_name), "TRR") \
                  for folder_name in low_folder_names_3d]


# ESMRMB
base_path_esmrmb_3d = r"C:\Users\Ivan\Documents\Sjukhusfysikerutbildning\Exjobb\Utvärdering av anpassningsmetoder\Parameter array\ESMRMB"
low_bvals = [100, 150, 200, 250, 300, 350, 400]
low_folder_names = [f"b abstract SNR[3, 6, 9, 12, 15, 18, 21] samples 50threshold{low_bval}" for low_bval in low_bvals]
esmrmb_low_dataset = [functions.FitMethod(os.path.join(base_path_esmrmb_3d, folder_name), "2-step Segmented") for folder_name in low_folder_names]
                  

#biexp_initial_opt = functions.FitMethod(path_initial_opt, "BiExp")

linear_initial_opt = functions.FitMethod(path_initial_opt, "Linear")
linear_no_covar_opt = functions.FitMethod(path_no_covar_opt, "Linear")
linear_covar_five = functions.FitMethod(path_covar_five, "Linear")
linear_generic = functions.FitMethod(path_generic, "Linear")
linear = [linear_initial_opt, linear_no_covar_opt, linear_covar_five, linear_generic]
linear2 = [linear_initial_opt, linear_generic]

sivim_initial_opt = functions.FitMethod(path_initial_opt, "sIVIM")
sivim_no_covar_opt = functions.FitMethod(path_no_covar_opt, "sIVIM")
sivim_covar_five = functions.FitMethod(path_covar_five, "sIVIM")
sivim_generic = functions.FitMethod(path_generic, "sIVIM")
sivim = [sivim_initial_opt, sivim_no_covar_opt, sivim_covar_five, sivim_generic]
sivim2 = [sivim_initial_opt, sivim_generic]

segmented_initial_opt = functions.FitMethod(path_initial_opt, "Segmented")
segmented_no_covar_opt = functions.FitMethod(path_no_covar_opt, "Segmented")
segmented_covar_five = functions.FitMethod(path_covar_five, "Segmented")
segmented_generic = functions.FitMethod(path_generic, "Segmented")
segmented = [segmented_initial_opt, segmented_no_covar_opt, segmented_covar_five, segmented_generic]
segmented2 = [segmented_initial_opt, segmented_generic]

segmented_2step_generic = functions.FitMethod(path_generic, "2-step segmented")

trr_initial_opt = functions.FitMethod(path_initial_opt, "TRR")
trr_no_covar_opt = functions.FitMethod(path_no_covar_opt, "TRR")
trr_covar_five = functions.FitMethod(path_covar_five, "TRR")
trr_generic = functions.FitMethod(path_generic, "TRR")
trr = [trr_initial_opt, trr_no_covar_opt, trr_covar_five, trr_generic]
trr2 = [trr_initial_opt, trr_generic]

varpro_initial_opt = functions.FitMethod(path_initial_opt, "VarPro")
varpro_no_covar_opt = functions.FitMethod(path_no_covar_opt, "VarPro")
varpro_covar_five = functions.FitMethod(path_covar_five, "VarPro")
varpro_generic = functions.FitMethod(path_generic, "VarPro")
varpro = [varpro_initial_opt, varpro_no_covar_opt, varpro_covar_five, varpro_generic]
varpro2 = [varpro_initial_opt, varpro_generic]

topopro_initial_opt = functions.FitMethod(path_initial_opt, "TopoPro")
topopro_no_covar_opt = functions.FitMethod(path_no_covar_opt, "TopoPro")
topopro_covar_five = functions.FitMethod(path_covar_five, "TopoPro")
topopro_generic = functions.FitMethod(path_generic, "TopoPro")
topopro = [topopro_initial_opt, topopro_no_covar_opt, topopro_covar_five, topopro_generic]
topopro2 = [topopro_initial_opt, topopro_generic]

data_initial_opt = [linear_initial_opt, sivim_initial_opt, segmented_initial_opt, trr_initial_opt, varpro_initial_opt, topopro_initial_opt]
data_initial_opt = [linear_generic, sivim_generic, segmented_generic, trr_generic, varpro_generic, topopro_generic]
data_all = [linear, sivim, segmented, trr, varpro, topopro]
data_all2 = [linear2, sivim2, segmented2, trr2, varpro2, topopro2]

SNR_table = functions.load_tables(path_initial_opt, "SNR_table")
b_values = functions.load_tables(path_initial_opt, "b_table")
parameter_table = functions.load_tables(path_initial_opt, "parameter_table")

b_sets = ["Initial opt", "Initial final", "No covar opt", "No covar final", "Covar five", "Covar fifty", "Generic"]
b_sets = ["Initial opt", "Initial final", "No covar opt", "Covar five", "Generic"]
b_sets = ["Const. s. t.", "Const. gen.", "NEX gen.", "Generic"]
b_sets2 = ["Const. s. t.", "Generic"]


#functions.plot_sum_rmse_vs_param(data_all2, parameter="f", SNR_idx=0, parameter_table=parameter_table, b_sets=b_sets2, save_name=os.path.join(path_report_figures, "sum_rmse_f_snr_3_clean"))
#functions.plot_sum_rmse_vs_param(data_all2, parameter="D*", SNR_idx=0, parameter_table=parameter_table, b_sets=b_sets2, save_name=os.path.join(path_report_figures, "sum_rmse_Dstar_snr_3_clean"))
#functions.plot_sum_rmse_vs_param(data_all2, parameter="D", SNR_idx=0, parameter_table=parameter_table, b_sets=b_sets2, save_name=os.path.join(path_report_figures, "sum_rmse_D_snr_3_clean"))

# f sum_rmse all
#functions.plot_sum_rmse_vs_param(data_all, parameter="f", SNR_idx=0, parameter_table=parameter_table, b_sets=b_sets, save_name=os.path.join(path_report_figures, "sum_rmse_f_snr_3"))
#functions.plot_sum_rmse_vs_param(data_all, parameter="f", SNR_idx=1, parameter_table=parameter_table, b_sets=b_sets, save_name=os.path.join(path_report_figures, "sum_rmse_f_snr_10"))
#functions.plot_sum_rmse_vs_param(data_all, parameter="f", SNR_idx=2, parameter_table=parameter_table, b_sets=b_sets, save_name=os.path.join(path_report_figures, "sum_rmse_f_snr_30"))

#functions.plot_sum_rmse_vs_param(data_all, parameter="D*", SNR_idx=0, parameter_table=parameter_table, b_sets=b_sets, save_name=os.path.join(path_report_figures, "sum_rmse_Dstar_snr_3"))
#functions.plot_sum_rmse_vs_param(data_all, parameter="D*", SNR_idx=1, parameter_table=parameter_table, b_sets=b_sets, save_name=os.path.join(path_report_figures, "sum_rmse_Dstar_snr_10"))
#functions.plot_sum_rmse_vs_param(data_all, parameter="D*", SNR_idx=2, parameter_table=parameter_table, b_sets=b_sets, save_name=os.path.join(path_report_figures, "sum_rmse_Dstar_snr_30"))

#functions.plot_sum_rmse_vs_param(data_all, parameter="D", SNR_idx=0, parameter_table=parameter_table, b_sets=b_sets, save_name=os.path.join(path_report_figures, "sum_rmse_D_snr_3"))
#functions.plot_sum_rmse_vs_param(data_all, parameter="D", SNR_idx=1, parameter_table=parameter_table, b_sets=b_sets, save_name=os.path.join(path_report_figures, "sum_rmse_D_snr_10"))
#functions.plot_sum_rmse_vs_param(data_all, parameter="D", SNR_idx=2, parameter_table=parameter_table, b_sets=b_sets, save_name=os.path.join(path_report_figures, "sum_rmse_D_snr_30"))

#functions.plot_f_bias_grid_vs_bvals(data_all, parameter="f", b_sets=b_sets, save_name=os.path.join(path_report_figures,"aaaaatest"))
#functions.plot_f_bias_grid_vs_bvals(data_all, parameter="D*", b_sets=b_sets)
#functions.plot_f_bias_grid_vs_bvals(data_all, parameter="D", b_sets=b_sets)
#functions.plot_rmse_grid_vs_protocol(data_all, parameter="D*", D_idx=3, SNR_idx=1, b_sets=b_sets)
#functions.plot_rmse_grid(data_initial_opt, parameter="f")#, save_name=os.path.join(path_report_figures, "rmse_grid_f_incl_biexp_initial_opt"))
#functions.plot_rmse_grid(data_initial_opt, parameter="D*")
#functions.plot_rmse_grid(data_initial_opt, parameter="D")

snr_levels = [3, 6, 9, 12, 15, 18, 21]
#functions.rmse_plot_3d(sivim_dataset_3d, "f", intermediary_bvals, snr_levels)
#functions.rmse_plot_3d(sivim_dataset_3d, "D", intermediary_bvals, snr_levels)
#functions.rmse_plot_3d(trr_dataset_3d, "D*", low_bvals, snr_levels)
#functions.rmse_plot_3d_all(sivim_dataset_3d, trr_dataset_3d, low_bvals, intermediary_bvals, snr_levels, save_name=os.path.join(path_report_figures, "3d_rmse_surfaces"))

#functions.rmse_plot_3d(esmrmb_low_dataset, "D*", low_bvals, snr_levels)
#functions.rmse_plot_3d(esmrmb_low_dataset, "f", low_bvals, snr_levels)
#functions.rmse_plot_3d(esmrmb_low_dataset, "D", low_bvals, snr_levels)
#functions.rmse_plot_3d_all_esmrmb(esmrmb_low_dataset, esmrmb_low_dataset, low_bvals, low_bvals, snr_levels, save_name=os.path.join(path_esmrmb_figures, "3d_rmse_surfaces"))

esmrmb_data = [[segmented_2step_generic], [varpro_generic], [topopro_generic]]
#functions.plot_f_bias_grid_vs_bvals(esmrmb_data, parameter="f", ylabels=["Segmented", "MIX", "TopoPro"], titles=["Bias in $f$"], figsize=(2.97,4.35), cbar_size=.2, fontsize=12, save_name=os.path.join(path_esmrmb_figures, "f bias"), filetype="png")
#functions.plot_f_bias_grid_vs_bvals(esmrmb_data, parameter="D", ylabels=["Segmented", "MIX", "TopoPro"], titles=["Bias in $D$"], figsize=(3.02,4.35), cbar_size=.2, fontsize=12, save_name=os.path.join(path_esmrmb_figures, "D bias"), filetype="png")
#functions.plot_f_bias_grid_vs_bvals(esmrmb_data, parameter="D*", ylabels=["Segmented", "MIX", "TopoPro"], titles=["Bias in $D^*$"], figsize=(2.87,4.35), cbar_size=.2, crop_Dstar=False, fontsize=12, save_name=os.path.join(path_esmrmb_figures, "Dstar bias"), filetype="png")

esmrmb_data = [segmented_2step_generic, varpro_generic, topopro_generic]
#functions.plot_rmse_grid_esmrmb(esmrmb_data, parameter="f", ylabels=["Segmented", "MIX", "TopoPro"], figsize=(3.99,4.35), cbar_size=.2, title="RMSE of $f$", fontsize=12, save_name=os.path.join(path_esmrmb_figures, "f RMSE SNR 3"))
#functions.plot_rmse_grid_esmrmb(esmrmb_data, parameter="D", ylabels=["Segmented", "MIX", "TopoPro"], figsize=(4.05,4.35), cbar_size=.2, title="RMSE of $D$", fontsize=12, save_name=os.path.join(path_esmrmb_figures, "D RMSE SNR 3"))
#functions.plot_rmse_grid_esmrmb(esmrmb_data, parameter="D*", ylabels=["Segmented", "MIX", "TopoPro"], figsize=(4,4.35), cbar_size=.2, title="RMSE of $D^*$", fontsize=12, crop_Dstar=False, save_name=os.path.join(path_esmrmb_figures, "Dstar RMSE SNR 3"))

for D_idx in range(len(parameter_table["D"])):
    fname_f = f"new_bias_map_f_D{D_idx}"
    fname_D_star = f"new_bias_map_Dstar_D{D_idx}"
    fname_D = f"new_bias_map_D_D{D_idx}"
    
    #fname_f = f"new_rmse_map_initial_opt_f_D{D_idx}"
    #fname_D_star = f"new_rmse_map_initial_opt_Dstar_D{D_idx}"
    #fname_D = f"new_rmse_map_initial_opt_D_D{D_idx}"
    
    functions.plot_f_bias_grid_vs_bvals(data_all, parameter="f", D_idx=D_idx, b_sets=b_sets, save_name=os.path.join(path_report_figures, fname_f))
    functions.plot_f_bias_grid_vs_bvals(data_all, parameter="D*", D_idx=D_idx, b_sets=b_sets, save_name=os.path.join(path_report_figures, fname_D_star))
    functions.plot_f_bias_grid_vs_bvals(data_all, parameter="D", D_idx=D_idx, b_sets=b_sets, save_name=os.path.join(path_report_figures, fname_D))
    
    #functions.plot_rmse_grid(data_initial_opt, parameter="f", D_idx=D_idx, save_name=os.path.join(path_report_figures, fname_f))
    #functions.plot_rmse_grid(data_initial_opt, parameter="D*", D_idx=D_idx, save_name=os.path.join(path_report_figures, fname_D_star))
    #functions.plot_rmse_grid(data_initial_opt, parameter="D", D_idx=D_idx, save_name=os.path.join(path_report_figures, fname_D))
