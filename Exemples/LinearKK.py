from utils import file_utils, linKK
#_______________________________________________________________________________________________________________________
# Read Data .CSV file
spec_obj = file_utils.read('../data/17 - EIS ferro(III) 5,0e-6 - DC 0,6 V.csv')

#_______________________________________________________________________________________________________________________
# Validate EIS data - by Linear Krammers Kroning
linkk_obj = linKK.LinearKramersKronig(spec_obj, c=0.1, max_iter=50, add_capacitor=True, verbose=True)
if linkk_obj.chi_square > 1e-2:
    raise ValueError(f'Linear Kramers-Kronig test failed: x² = {linkk_obj.chi_square}')
