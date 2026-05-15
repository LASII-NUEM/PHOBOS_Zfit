from utils import data_types, file_utils, fitting_utils
import numpy as np
from impedance.validation import linKK
import matplotlib.pyplot as plt

class LinearKramersKronig:
    def __init__(self, data_medium:data_types.SpectroscopyData, c=0.5, max_iter=100, add_capacitor=True, verbose=False):
        '''
        :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
        :param c: threshold for overfitting criterion (mu)
        :param max_iter: maximum number of M RC pairs
        :param add_capacitor: flag to add a capacitor in series to the R_ohm+L+R_k block
        :param verbose: flag to print the statistics in the terminal
        '''

        #validate data_medium
        expected_types = [data_types.SpectroscopyData, list]
        if type(data_medium) not in expected_types:
            raise TypeError(f'[LinearKramersKronig] "data_medium" must be {expected_types}! Curr. type = {type(data_medium)}')

        self.c = c
        self.add_capacitor = add_capacitor
        self.max_iter = max_iter
        self.fit_type = 'complex'
        self.freqs = data_medium.freq
        self.z_meas_real = data_medium.Z_real
        self.z_meas_imag = data_medium.Z_imag

        f_Hz = np.asarray(self.freqs, float).ravel()
        z_meas = data_medium.Z_real - 1j * data_medium.Z_imag  # complex impedance
        Z_meas_arr = np.atleast_2d(z_meas)

        for i in range(Z_meas_arr.shape[0]):

            print(f'[LinerKramersKronig] \nLinKK test for: {data_medium.sheet_names[i]}')
            self.Z_meas = np.asarray(Z_meas_arr[i,:], dtype=complex).ravel()
            # M (chosen number of RC elements), Z_fit (KK-consistent), res_real, res_imag
            self.M,  self.mu,  self.Z_fit,  self.res_real,  self.res_imag = linKK(f_Hz, self.Z_meas, self.c, self.max_iter, self.fit_type, self.add_capacitor)
            self.chi_square = self.compute_chi_square()

            # handy scalar metrics
            self.rms_rel_re = np.sqrt(np.mean( self.res_real ** 2))
            self.rms_rel_im = np.sqrt(np.mean( self.res_imag ** 2))

            #residue test < 1%

            self.plotLinkk(title=data_medium.sheet_names[i])

            if verbose:
                print(f'[LinerKramersKronig] Data validity test for {data_medium.sheet_names[i]}:')
                if self.chi_square > 1e-2:
                    print(f': Linear Kramers-Kronig test failed: x² = {self.chi_square}')
                    print()
                else:
                    print(f'Linear Kramers-Kronig test passed: x² = {self.chi_square}')
                    print(f'Optimal M RC components = {self.M}')
                    print()

    def compute_chi_square(self):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: chi square of the fit
        '''

        #validate 'z'
        if not isinstance(self.Z_meas, np.ndarray):
            raise TypeError(f'[LinearKramersKronig] "z" must be a Numpy Array! Curr. type = {type(self.Z_meas)}')

        #validate 'z_hat'
        if not isinstance(self.Z_fit, np.ndarray):
            raise TypeError(f'[LinearKramersKronig] "z_hat" must be a Numpy Array! Curr. type = {type(self.Z_fit)}')

        #validate shape
        if len(self.Z_meas) != len(self.Z_fit):
            raise ValueError(f'[LinearKramersKronig] "z" and "z_hat" must match in length!')

        chi_square_real = ((self.Z_meas.real-self.Z_fit.real)/np.abs(self.Z_meas))**2
        chi_square_imag = ((self.Z_meas.imag-self.Z_fit.imag)/np.abs(self.Z_meas))**2
        chi_square = (chi_square_real+chi_square_imag)/(self.Z_meas.real**2 + self.Z_meas.imag**2)

        return np.sum(chi_square)

    def plotLinkk(self, title=None):

        freqs =  np.asarray(self.freqs, float).ravel()
        Z_meas = np.asarray(self.Z_meas, complex).ravel()
        Z_fit = np.asarray(self.Z_fit, complex).ravel()
        res_re = np.asarray(self.res_real, float).ravel()
        res_im = np.asarray(self.res_imag, float).ravel()
        # ---- Nyquist ----

        plt.figure(figsize=(7, 6))
        plt.plot(Z_meas.real, -Z_meas.imag, "o", label="Measured")
        plt.plot(Z_fit.real, -Z_fit.imag, "-", label=f"KK fit (M={self.M})")
        plt.xlabel("Z' [Ω]")
        plt.ylabel("-Z'' [Ω]")
        plt.title(f"{title}: Nyquist - Fit Linear Kramers-Kronig")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ---- Residuals ----
        plt.figure(figsize=(8, 5))
        plt.semilogx(freqs, res_re, "o-", label="Residual Re (relative)")
        plt.semilogx(freqs, res_im, "s-", label="Residual Im (relative)")
        plt.text(0.95, 0.95,
                 f"$\\chi^2$ = {self.chi_square:.2e}",
                 transform=plt.gca().transAxes,
                 ha="right",
                 va="top",
                 fontsize=12,
                 bbox=dict(boxstyle="round",
                           ec="red",
                           fc="white",
                           alpha=0.8)
                 )
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Relative residual [-]")
        plt.title(f"{title}: Residuals - Linear Kramers-Kronig")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.show()