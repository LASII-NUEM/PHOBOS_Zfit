from utils import ECM_utils
import numpy as np
#_______________________________________________________________________________________________________________________
# Defining the fitting parameters from the chosen Electric Circuit Model (ECM)

# write the circuit as text
circuit= "(C//R)"

# define circuit params
ECM_Params = ECM_utils.CircuitParams(circuit, Zequation = True)

print("Circuit:")
print(ECM_Params.circuit)
print("Parameter names:")
print(ECM_Params.param_names[:])

#_______________________________________________________________________________________________________________________
# Evaluate the frequency response of the impedance for the ECM

# define frequency values
freqs = np.logspace(np.log10(40), np.log10(1e6), 210)

# define parameter values
param_value = np.array([1000.0,1e-9])
scaling = np.array([1,1])

ECM_Z = ECM_utils.CircuitEvaluate(freqs, ECM_Params, param_value, scaling, verbose=True)