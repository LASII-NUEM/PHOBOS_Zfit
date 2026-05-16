from utils import file_utils

# To upload file .csv
spec_obj_csv = file_utils.read('../data/17 - EIS ferro(III) 5,0e-6 - DC 0,6 V.csv', flip=True)

# or upload file .xlsx
spec_obj_xlsx = file_utils.read('../data/Dados Everton.xlsx', flip=True)

# just change the file repository.
# flip argument is for descending or crescent values

# to reach the values just called the obj created and the parameter
freqs_csv= spec_obj_csv.freq
z_real_csv = spec_obj_csv.Z_real
z_imag_csv = spec_obj_csv.Z_imag


