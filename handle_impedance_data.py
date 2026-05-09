from utils import file_utils, linKK, ECM_utils
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#_______________________________________________________________________________________________________________________
# READ DATA

spec_obj = file_utils.read('./data/17 - EIS ferro(III) 5,0e-6 - DC 0,6 V.csv')
freqs = spec_obj.freq
z_real = spec_obj.Z_real
z_imag = -spec_obj.Z_imag #imaginary impedance is negative

#_______________________________________________________________________________________________________________________
# PLOT DATA

fig = plt.figure(figsize=(8,6) )
plt.plot(z_real, z_imag, 'b-')
plt.title('Nyquist Impedance')
plt.ylabel("Z'")
plt.xlabel("Z''")
plt.grid(True)
plt.tight_layout()

Z_real = np.asarray(z_real, dtype=float)
Z_imag = np.asarray(z_imag, dtype=float)
Z_mag = np.sqrt(Z_real**2 + Z_imag**2)
Z_phase = np.degrees(np.arctan2(Z_imag, Z_real))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle('Impedance Data')
# Bode magnitude
ax1.plot(freqs, Z_mag, color='blue')
ax1.set_xscale('log')
ax1.set_ylabel('|Z| (Ohm)')
ax1.set_title('Bode Plot')
ax1.grid(True)
# Phase
ax2.plot(freqs, Z_phase, color='red')
ax2.set_xscale('log')
ax2.set_ylabel('Phase (deg)')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_title('Phase Plot')
ax2.grid(True)
plt.tight_layout()
plt.show()

#_______________________________________________________________________________________________________________________
# Validate EIS data - by Linear Krammers Kroning

linkk_obj = linKK.LinearKramersKronig(spec_obj, c=0.1, max_iter=50, add_capacitor=True, verbose=False)

if linkk_obj.chi_square > 1e-2:
    raise ValueError(f'Linear Kramers-Kronig test failed: x² = {linkk_obj.chi_square}')

#_______________________________________________________________________________________________________________________
# Defining the fitting parameters from the choosen Electric Circuit Model (ECM)

# write the circuit as text
circuit = "(R//(R+L+C))+(C//W)"

ECM_Params = ECM_utils.CircuitParams(circuit)

print("Circuit:")
print(circuit)
print("Parameter names:")
print(ECM_Params.param_names[:])


#_______________________________________________________________________________________________________________________
# Evaluate the frequency response of the impedance for the ECM

# define parameter values
# params = {
#     "R1": 100.0,
#     "R2": 300.0,
#     "L1":1e-6,
#     "C1": 1e-6,
#     "C2": 1e-6,
#     "Q1": 2e-4,
#     "alpha1": 0.85,
#     "W1": 10.0
# }

# ECM_Z = ECM_utils.CircuitEvaluate(freqs, params, ECM_Params.tree)

#_______________________________________________________________________________________________________________________
#