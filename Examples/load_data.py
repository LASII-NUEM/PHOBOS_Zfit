from utils import file_utils

# To upload EIS file .csv
spec_obj_csv = file_utils.read('../data/17 - EIS ferro(III) 5,0e-6 - DC 0,6 V.csv', type = "EIS")

# or upload EIS file .xlsx
# spec_obj_xlsx = file_utils.read('../data/Dados Everton.xlsx', type = "EIS")

#  or upload phobos .xlsx
# spec_obj_phobos = file_utils.read('../data/phobos_cc.csv', type = "PHOBOS")
# spec_obj_phobos = file_utils.read('../data/phobos_multielectrode.csv', type = "PHOBOS")

# to reach the values just called the obj created and the parameter
freqs_csv= spec_obj_csv.freq
z_real_csv = spec_obj_csv.Z_real
z_imag_csv = spec_obj_csv.Z_imag