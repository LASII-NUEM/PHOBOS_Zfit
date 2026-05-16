from utils import file_utils, linKK, ECM_utils, fitting_utils
import numpy as np
import matplotlib.pyplot as plt

#user input variables
filename = '../data/Dados Everton.xlsx'  #name of the file with the acquired data
circuit= "(R1 +(R2//CPE1)+(R3//CPE2))" #string with the circuit description code
initial_guess = np.array([1,1,1,0.5,1,1,0.5]) #initial guess for the fitting routine
scaling_array = np.array([[1e3, 1e2, 1e-4, 1, 1e3, 1e-4, 1],
                         [1e3, 1e3, 1e-6, 1, 1e4, 1e-1, 1],
                         [1e3, 1e3, 1e-5, 1, 1e3, 1e-3, 1],
                         [1e3, 1e3, 1e-6, 1, 1e3, 1e-5, 1]]) #scaling array for each parameter and each dataset
method = 'BFGS' #which estimator will be used to compute the fit
plot_style = 'tiles' #how to display the fitted data (i.e., 'tile' divides into subplots)

#read the .xlsx data and store in a dedicated data structure
spec_obj = file_utils.read(filename, flip=True)
Z_real = np.asarray(spec_obj.Z_real, dtype=float)
Z_imag = np.asarray(spec_obj.Z_imag, dtype=float)
Z_complex = (Z_real-1j*Z_imag).astype("complex")
Z_mag = np.abs(Z_complex)
Z_phase = np.rad2deg(np.angle(Z_complex))

#define the fitting parameters from the input Electric Circuit Model (ECM)
ECM_Params = ECM_utils.CircuitParams(circuit)
print("[fit_nyquist] Circuit:")
print(ECM_Params.circuit)
print("[fit_nyquist] Parameters:")
print(ECM_Params.param_names[:])

#circuit fitting routine
ECM_fit_obj = fitting_utils.Circuit_fitting(spec_obj,spec_obj.freq,ECM_Params)
ECM_fit_params = ECM_fit_obj.fit_circuit(initial_guess, scaling_array, method=method, verbose=True)
Z_bfgs = ECM_fit_params.opt_fit

#plot the results
valid_syles = ['tiles', 'individual']
if plot_style not in valid_syles:
    raise ValueError(f'[fit_nyquist] {plot_style} style plot is a valid type! Try: {valid_syles}')
if plot_style == 'individual':
    for i in range(len(Z_real[:,0])):
        fig, ax = plt.subplots(figsize=(12,10))
        plt.title(f'[Nyquist] BFGS/NLLS Fit: {spec_obj.sheet_names[i]}')
        leg = []
        ax.plot(Z_bfgs[i].Z_ECM.real, -Z_bfgs[i].Z_ECM.imag, color="tab:blue")
        leg.append(f'BFGS: χ2 = {ECM_fit_params.chi_square[i]}')
        ax.scatter(Z_real[i,:], Z_imag[i,:], color="black")
        leg.append('measured')

        plt.xlabel("Z'")
        plt.ylabel("Z''")
        plt.legend(leg)
        plt.grid()
        plt.show()

elif plot_style == 'tiles':
    plt.figure()
    n_plots = len(Z_bfgs) #number of computed spectra
    if n_plots%2 == 0:
        n_lines = int(n_plots/2)
        n_cols = int(n_plots/2)
        for line in range(0,n_lines):
            for col in range(0, n_cols):
                plot_idx = n_cols*line+col
                curr_data = Z_bfgs[plot_idx]
                plt.subplot(n_lines, n_cols, plot_idx+1)
                plt.title(f'{spec_obj.sheet_names[plot_idx]}')
                plt.plot(curr_data.Z_ECM.real, -curr_data.Z_ECM.imag, color="red", label=f'BFGS: χ2 = {ECM_fit_params.chi_square[plot_idx]}')
                plt.scatter(Z_real[plot_idx,:], Z_imag[plot_idx,:], color="black", label='measured')
                plt.xlabel("R(Z) [Ω]")
                plt.ylabel("-I(Z) [Ω]")
                plt.legend()
                plt.grid()
        plt.show()
    else:
        for plot_idx in range(0,n_plots):
            plt.subplot(1,n_plots,plot_idx+1)
            curr_data = Z_bfgs[plot_idx]
            plt.title(f'{spec_obj.sheet_names[plot_idx]}')
            plt.plot(curr_data.Z_ECM.real, -curr_data.Z_ECM.imag, color="red", label=f'BFGS: χ2 = {ECM_fit_params.chi_square[plot_idx]}')
            plt.scatter(Z_real[plot_idx, :], Z_imag[plot_idx, :], color="black", label='measured')
            plt.xlabel("R(Z) [Ω]")
            plt.ylabel("-I(Z) [Ω]")
            plt.legend()
            plt.grid()
        plt.show()


