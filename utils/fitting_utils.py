from utils import data_types, ECM_utils
import numpy as np
from scipy.optimize import curve_fit, minimize, least_squares
import time

class Circuit_fitting:
    def __init__(self, data_medium:data_types.SpectroscopyData, freqs:np.ndarray, ecm: ECM_utils.CircuitParams):
        '''
        :param data_medium: the medium of the spectroscopic data
        :param freqs: the frequencies of the spectroscopic data
        :param ECM: the Equivalent Circuit Model defined by user
        '''

       #validate data_medium
        expected_Datatypes = [data_types.SpectroscopyData, list]
        if type(data_medium) not in expected_Datatypes:
            raise TypeError(f'[EquivalentCircuit] "data_medium" must be {expected_Datatypes}! Curr. type = {type(data_medium)}')

        self.data_medium = data_medium
        self.z_meas_real = data_medium.Z_real
        self.z_meas_imag = data_medium.Z_imag
        self.z_meas = self.z_meas_real - 1j * self.z_meas_imag  # complex impedance

        #validate freqs
        if not isinstance(freqs, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')
        self.freqs = freqs.astype(float)

       #validate ECM data
        expected_ECMtypes = [ECM_utils.CircuitParams, list]
        if type(ecm) not in expected_ECMtypes:
            raise TypeError(f'[EquivalentCircuit] "ECM" must be {expected_ECMtypes}! Curr. type = {type(data_medium)}')
        self.ecm = ecm

        self.scaling = None
        self.fit_method_reponse = []
        self.opt_params = []
        self.opt_scaled_params = []
        self.opt_cost = []
        self.opt_fitting = []
        self.error_NMSE = []
        self.error_NRMSE = []
        self.error_CHISQR = []
        self.error_MAE = []
        self.inter_num = []
        self.fit_elapsed_time = []

    def fit_circuit(self, initial_guess:np.ndarray, scaling_array:np.ndarray, method='BFGS', tol=1e-6, verbose=False):
        '''
        :param initial_guess: the initial guess for the fit to run the iterative algorithms
        :param scaling_array: scale all the search parameters to avoid exploding gradients
        :param method: which optimization algorithm will be used to fit the circuit data
        :param tol: tolerance for the algorithms
        :param verbose: flag to print the statistics in the terminal
        :return: the parameters the best fit the expected equivalent circuit
        '''

        #validate "method"
        valid_methods = ['BFGS', 'NLLS']
        if method not in valid_methods:
            raise ValueError(f'[EquivalentCircuit] {method} not implemented! Try: {valid_methods}')
        self.fit_method = method

        #validate "initial_guess"
        if len(initial_guess) != len(self.ecm.param_names):
            raise ValueError(f'[EquivalentCircuit] The number of initial guess parameters do not match with the given model! Should be {len(self.ecm.param_names)}.')

        if not isinstance(initial_guess, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "initial_guess" must be a Numpy Array! Curr. type = {type(initial_guess)}')

        #validate "scaling_array"
        Z_meas_arr = np.atleast_2d(self.z_meas)
        if len(scaling_array[0,:]) != len(initial_guess):
            raise ValueError(f'[EquivalentCircuit] Length of the scaling array do not match the length of the initial guess! {len(scaling_array)} != {len(initial_guess)}')
        if scaling_array.shape[0] != Z_meas_arr.shape[0]:
            raise ValueError(f'[EquivalentCircuit] The amount of scaling arrays do not match the amount of spectra! {scaling_array.shape[0]} != {Z_meas_arr.shape[0]}')
        self.scaling = scaling_array

        if not isinstance(scaling_array, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "scaling_array" must be a Numpy Array! Curr. type = {type(scaling_array)}')

        bounds = self.ecm.bound #optimization boundaries
        if self.fit_method == "BFGS":
            for i in range(Z_meas_arr.shape[0]):
                z_raw = Z_meas_arr[i, :]
                t_init = time.time()
                min_obj = minimize(self.CUMSE, initial_guess, args=([z_raw.astype('complex'), self.freqs, scaling_array[i,:]]), bounds=bounds, method='L-BFGS-B')
                t_elapsed = time.time() - t_init
                opt_fit = ECM_utils.CircuitEvaluate(self.freqs, self.ecm, min_obj.x, scaling_array[i,:], verbose=False)
                Z_fit = opt_fit.Z_ECM
                opt_params_scaled = min_obj.x*scaling_array[i,:] #rescale the minimized parameters
                nmse = self.NMSE(z_raw.astype("complex"), Z_fit.astype('complex')) #NMSE score for both complex parts
                nrmse = self.NRMSE(z_raw.astype("complex"), Z_fit.astype('complex')) #nrmse score for both complex parts
                chisqr = self.chi_square(z_raw.astype("complex"), Z_fit.astype('complex')) #chi-square score for both complex parts
                mae = self.MAE(z_raw.astype("complex"), Z_fit.astype('complex')) #mae score for both complex parts

                if verbose:
                    print(f'[EquivalentCircuit] Quasi-Newton-based impedance fitting:')
                    print(f'Test name: {self.data_medium.sheet_names[i]}')
                    print(f't = {t_elapsed} s')
                    print(f'NMSE = {nmse}')
                    print(f'NRMSE = {nrmse}')
                    print(f'chi-square = {chisqr}')
                    print(f'MAE = {mae}')
                    fit_params = self.ecm.param_names
                    print(f'fitted params = ')
                    for i in range(len(fit_params)):
                        print(f'{fit_params[i]} = {opt_params_scaled[i]}')
                    print()

                self.fit_method_reponse.append(min_obj)
                self.opt_params.append(opt_fit.params_value)
                self.opt_scaled_params.append(opt_params_scaled)
                self.opt_cost.append(min_obj.fun)
                self.opt_fitting.append(opt_fit)
                self.error_NMSE.append(nmse)
                self.error_NRMSE.append(nrmse)
                self.error_CHISQR.append(chisqr)
                self.error_MAE.append(mae)
                self.inter_num.append(min_obj.nit)
                self.fit_elapsed_time.append(t_elapsed)

            return OptimizerResults(fit_result = self.fit_method_reponse, opt_params= self.opt_params, opt_params_scaled=self.opt_scaled_params,
                                    opt_cost= self.opt_cost,
                                    opt_fit=self.opt_fitting, nmse_score=self.error_NMSE, nrmse_score=self.error_NRMSE, chi_square=self.error_CHISQR,
                                    mae_score=self.error_MAE,
                                    n_iter= self.inter_num, t_elapsed=self.fit_elapsed_time)  # return the optimized parameters

        elif self.fit_method == "NLLS":
            bounds = np.array(bounds)  # convert the boundaries to numpy array
            bounds = ((bounds[:, 0]), (bounds[:, 1]))  # curve_fit receives bounds as tuple
            for i in range(Z_meas_arr.shape[0]):
                z_raw = Z_meas_arr[i, :]
                t_init = time.time()
                ls_obj = least_squares(self.CUMSE, x0=initial_guess, args=([[z_raw.astype('complex'), self.freqs, scaling_array[i,:]]]),
                                       bounds=bounds, max_nfev=5000) #Trust-Region-based NLLS
                t_elapsed = time.time() - t_init
                opt_fit = ECM_utils.CircuitEvaluate(self.freqs, self.ecm, ls_obj.x, scaling_array[i,:], verbose=False)
                Z_fit = opt_fit.Z_ECM
                opt_params_scaled = ls_obj.x*scaling_array[i,:] #rescale the minimized parameters
                nmse = self.NMSE(z_raw.astype("complex"), Z_fit.astype("complex")) #NMSE score for both complex parts
                nrmse = self.NRMSE(z_raw.astype("complex"), Z_fit.astype("complex")) #nrmse score for both complex parts
                chisqr = self.chi_square(z_raw.astype("complex"), Z_fit.astype("complex")) #chi-square score for both complex parts
                mae = self.MAE(z_raw.astype("complex"), Z_fit.astype("complex"))  # mae score for both complex parts

                if verbose:
                    print(f'[EquivalentCircuit] Non-linear least squares impedance fitting:')
                    print(f'Test name: {self.data_medium.sheet_names[i]}')
                    print(f'NLLS fit elapsed time = {t_elapsed} s')
                    print(f'NMSE = {nmse}')
                    print(f'NRMSE = {nrmse}')
                    print(f'chi-square = {chisqr}')
                    print(f'MAE = {mae}')
                    fit_params = self.ecm.param_names
                    print(f'fitted params = ')
                    for i in range(len(fit_params)):
                        print(f'{fit_params[i]} = {opt_params_scaled[i]}')
                    print()

                self.fit_method_reponse.append(ls_obj)
                self.opt_params.append(opt_fit.params_value)
                self.opt_scaled_params.append(opt_params_scaled)
                self.opt_fitting.append(opt_fit)
                self.error_NMSE.append(nmse)
                self.error_NRMSE.append(nrmse)
                self.error_CHISQR.append(chisqr)
                self.error_MAE.append(mae)
                self.fit_elapsed_time.append(t_elapsed)

            return OptimizerResults(fit_result= self.fit_method_reponse, opt_params= self.opt_params, opt_params_scaled= self.opt_scaled_params,
                                        opt_fit= self.opt_fitting, nmse_score=self.error_NMSE, nrmse_score=self.error_NRMSE,
                                    chi_square=self.error_CHISQR, mae_score= self.error_MAE, t_elapsed= self.fit_elapsed_time)

        else:
            raise ValueError(f'[EquivalentCircuit] method = {method} not implemented! Try: {valid_methods}')

    def CUMSE(self, theta, args):
        '''
        :param z_hat: the complex impedance computed from the fitted circuit
        :param z_meas: the complex impedance measured from the real system
        :return: the mean squared error between the measured and fitted impedance values
        '''

        ecm_hat = ECM_utils.CircuitEvaluate(args[1], self.ecm, theta, args[2], verbose=False)
        z_hat = ecm_hat.Z_ECM
        z_hat = z_hat.astype('complex')
        args[0] = args[0].astype('complex')
        if z_hat.ndim >= 2:
            z_hat = z_hat.T
            SSE = np.sum(((args[0].real-z_hat.real)**2)+((args[0].imag-z_hat.imag)**2), axis=1)
        else:
            SSE = np.sum(((args[0].real-z_hat.real)**2) + ((args[0].imag-z_hat.imag)**2))

        return SSE / len(z_hat)

    def NRMSE(self, z: np.ndarray, z_hat: np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: NRMSE of the fit for both real and imaginary part
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        #mean absolute percentage error
        SSE = np.sum(((z_hat.real-z.real)**2)+((z_hat.imag-z.imag)**2)) #sum of squared errors
        RMSE = np.sqrt(SSE/len(z)) #root mean squared error
        CMEAN = np.mean(z.real+z.imag) #complex mean

        return np.abs(RMSE/CMEAN)

    def NMSE(self, z:np.ndarray, z_hat:np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: NMSE of the fit
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        #normalized mean squared error
        SSE = np.sum(((z_hat.real-z.real)**2) + ((z_hat.imag-z.imag)**2)) #sum of squared errors
        SSO = np.sum((z.real**2) + (z.imag**2)) #sum of squared measurements

        return SSE/SSO

    def chi_square(self, z:np.ndarray, z_hat:np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: chi square of the fit
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        W = np.abs(1/(z.T @ z))
        W = np.abs(W)*np.ones(len(z)) #weighting matrix
        res_real = z.real-z_hat.real #real residue
        chi_square_real = res_real.T @ (W*res_real)
        res_imag = z.imag-z_hat.imag #imaginary residue
        chi_square_imag = res_imag.T @ (W*res_imag)

        return chi_square_real + chi_square_imag

    def MAE(self, z:np.ndarray, z_hat:np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: chi square of the fit
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        z_abs = np.abs(z)
        z_hat_abs = np.abs(z_hat)
        abs_err = np.abs(z_abs-z_hat_abs)/z_abs #absolute error

        return 100*np.mean(abs_err)

class OptimizerResults:
    def __init__(self, fit_result=None, opt_params=None, opt_params_scaled=None, opt_cost=None, opt_fit=None, nmse_score=None, nrmse_score=None, chi_square=None, mae_score=None, n_iter=None, t_elapsed=None):
        if fit_result is not None:
            self.fit_reponse = fit_result
        if opt_params is not None:
            self.opt_params = opt_params
        if opt_params_scaled is not None:
            self.opt_params_scaled = opt_params_scaled
        if opt_cost is not None:
            self.opt_cost = opt_cost
        if opt_fit is not None:
            self.opt_fit = opt_fit
        if nmse_score is not None:
            self.nmse_score = nmse_score
        if nrmse_score is not None:
            self.nrmse_score = nrmse_score
        if chi_square is not None:
            self.chi_square = chi_square
        if mae_score is not None:
            self.mae_score = mae_score
        if n_iter is not None:
            self.n_iter = n_iter
        if t_elapsed is not None:
            self.t_elapsed = t_elapsed