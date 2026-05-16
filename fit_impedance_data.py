from utils import file_utils, linKK, ECM_utils, fitting_utils
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#_______________________________________________________________________________________________________________________
# READ DATA .XLSX
spec_obj = file_utils.read('./data/Dados Everton.xlsx')
freqs = spec_obj.freq
z_real = spec_obj.Z_real
z_imag = spec_obj.Z_imag #imaginary impedance is negative

Z_real = np.asarray(z_real, dtype=float)
Z_imag = np.asarray(z_imag, dtype=float)
Z_mag = np.sqrt(Z_real**2 + Z_imag**2)
Z_phase = np.degrees(np.arctan2(Z_imag, Z_real))

#_______________________
# PLOT DATA
label=["65 10 umolL - Pb S25", "65 10 umolL - Cd S15", "65 10 umolL - Hg S8", "65 Tampão Hh S1"]
fig = plt.figure(figsize=(8,6))
for i in range(len(z_real[:,0])):
    plt.plot(z_real[i,:], z_imag[i,:], label=label[i])
plt.title('Nyquist Impedance')
plt.ylabel("Z'")
plt.xlabel("Z''")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle('Impedance Data')

# Bode magnitude
for i in range(len(Z_mag[:,0])):
    ax1.plot(freqs, Z_mag[i,:], label=label[i])
ax1.set_xscale('log')
ax1.set_ylabel('|Z| (Ohm)')
ax1.set_title('Bode Plot')
ax1.legend(loc='upper right')
ax1.grid(True)

# Phase
for i in range(len(Z_phase[:,0])):
    ax2.plot(freqs, Z_phase[i,:], label=label[i])
ax2.set_xscale('log')
ax2.set_ylabel('Phase (deg)')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_title('Phase Plot')
ax2.grid(True)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

#_______________________________________________________________________________________________________________________
# Validate EIS data - by Linear Krammers Kroning

linkk_obj = linKK.LinearKramersKronig(spec_obj, c=0.1, max_iter=50, add_capacitor=True, verbose=True)
if linkk_obj.chi_square > 1e-2:
    raise ValueError(f'Linear Kramers-Kronig test failed: x² = {linkk_obj.chi_square}')

#_______________________________________________________________________________________________________________________
# Defining the fitting parameters from the chosen Electric Circuit Model (ECM)

# write the circuit as text
circuit= "(R1 +(R2//CPE1)+(R3//CPE2))" # prof paula dados everton .xlsx
ECM_Params = ECM_utils.CircuitParams(circuit, Zequation = True)

print("Circuit:")
print(ECM_Params.circuit)
print("Parameter names:")
print(ECM_Params.param_names[:])
#_______________________________________________________________________________________________________________________
# fitting data for .xlsx file
# method = "BFGS" or "NLLS"

ECM_fit = fitting_utils.Circuit_fitting(spec_obj,freqs,ECM_Params)

initial_params = np.array([1,1,1,0.5,1,1,0.5]) #initial guess for the fitting routine
scaling_array = np.array([[1e3, 1e2, 1e-4, 1, 1e3, 1e-4, 1],
                         [1e3, 1e3, 1e-6, 1, 1e4, 1e-1, 1],
                         [1e3, 1e3, 1e-5, 1, 1e3, 1e-3, 1],
                         [1e3, 1e3, 1e-6, 1, 1e3, 1e-5, 1]])

ECM_BFGS_params = ECM_fit.fit_circuit(initial_params, scaling_array, method="BFGS", verbose=False)
ECM_NLLS_params = ECM_fit.fit_circuit(initial_params, scaling_array, method="NLLS", verbose=False)

#________________________
# Plot fitting result

Z_bfgs = ECM_BFGS_params.opt_fit
Z_nlls = ECM_NLLS_params.opt_fit

for i in range(len(Z_real[:,0])):
    fig, ax = plt.subplots(figsize=(12,10))
    plt.title(f'[Nyquist] BFGS/NLLS Fit: {spec_obj.sheet_names[i]}')

    ax.plot(Z_bfgs[i].Z_ECM.real, -Z_bfgs[i].Z_ECM.imag, color="blue", label=f'BFGS: χ2 = {ECM_BFGS_params.chi_square[i]}')
    ax.plot(Z_nlls[i].Z_ECM.real, -Z_nlls[i].Z_ECM.imag, color="red",  label=f'NLLS: χ2 = {ECM_NLLS_params.chi_square[i]}')
    ax.scatter(Z_real[i,:], Z_imag[i,:], color="black", label='measured')

    ax.set_xlabel("Z' [Ω]")
    ax.set_ylabel("-Z'' [Ω]")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()
