from utils import file_utils
import numpy as np
import matplotlib.pyplot as plt

spec_obj = file_utils.read('../data/17 - EIS ferro(III) 5,0e-6 - DC 0,6 V.csv', flip=True)
freqs = spec_obj.freq
z_real = spec_obj.Z_real
z_imag = -spec_obj.Z_imag #imaginary impedance is negative

#_______________________
# PLOT DATA

#  Nyquist Plot
fig = plt.figure(figsize=(8,6))
plt.plot(z_real, z_imag, 'b-')
plt.title('Nyquist Impedance')
plt.ylabel("Z'")
plt.xlabel("Z''")
plt.grid(True)
plt.tight_layout()
plt.show()

Z_real = np.asarray(z_real, dtype=float)
Z_imag = np.asarray(z_imag, dtype=float)
Z_mag = np.sqrt(Z_real**2 + Z_imag**2)
Z_phase = np.degrees(np.arctan2(Z_imag, Z_real))

#  Bode/Phase Plot

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