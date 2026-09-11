from utils import data_types
import pandas as pd
import csv
import numpy as np
import os


def read(filename: str, type: str):
    '''
    :param filename: path where the .csv is stored
    :param flip: flag to flip the matrices and arrays
    '''

    # check if the filename exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'[file_utils] Filename {filename} does not exist!')

    file_ext = [".csv", ".xls", ".xlsx"]

    ext = os.path.splitext(filename)[1].lower()
    if ext not in file_ext:
        raise TypeError(f'[file_utils] Unknown file extension! Curr. type = {ext}')

    file_type = ["eis", "phobos"]
    Type = type.lower()
    if Type not in file_type:
        raise TypeError(f'[file_utils] Unknown file type! Curr. type = {Type}')

    sheets = []
    if Type == "eis":
        if ext == ".csv":
            # process the raw data output  into a custom data structure
            raw_data = pd.read_csv(
                filename,
                delimiter=",",
                skiprows=5,
                encoding="utf-16").to_numpy()

            base = os.path.basename(filename)
            file, ext = os.path.splitext(base)
            sheets.append(file)
            raw_data_all = raw_data[:, :-1]
            n_freqs = get_n_freqs(filename)
            freqs = raw_data_all[:, 0]

        elif ext in [".xls", ".xlsx"]:
            raw_data = pd.ExcelFile(filename)
            sheet_names = raw_data.sheet_names
            n_sheets = len(sheet_names)
            raw_data_all = []

            for i in range(n_sheets):
                raw_data = pd.read_excel(
                    filename,
                    sheet_name=i
                ).to_numpy()

                if raw_data.size == 0:
                    print(f"[file_utils] Sheet {sheet_names[i]} is empty.")
                    continue
                sheets.append(sheet_names[i])
                raw_data = raw_data[1:, :]
                raw_data_all.append(raw_data)

            raw_data_all = np.array(raw_data_all)
            freqs = raw_data_all[0, :, 0]
            n_freqs = len(raw_data_all[0, :, 0])

        else:
            raise TypeError(f'[filew_utils] Unknown file type! Curr. type = {ext}')
    elif Type == "phobos":

        # infer the swept frequencies from the file header
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)  # read only the header
            header_data = reader.fieldnames
        f.close()  # close the file

        # process the raw data output from the PHOBOS acquisition system into a custom data structure
        raw_data_all = pd.read_csv(
            filename).to_numpy()  # process the raw data output from the PHOBOS acquisition system
        freqs = np.array([float(freq.replace(" ", "").replace("Cp", "").replace("Z", ""))
                          for freq in header_data if "Cp" in freq or "Z" in freq])
        n_freqs = len(freqs)

    else:
        raise TypeError(f'[file_utils] Unknown file type! Curr. type = {Type}')

    data = data_types.SpectroscopyData(raw_data_all, freqs, n_freqs, ext, Type, sheets)

    return data

def get_n_freqs(filepath):
    with open(filepath, 'r', encoding='utf-16') as f:
        for line in f:
            if 'freqs' in line.lower():
                # Extract number before "freqs"
                n_freqs = int(line.strip().split()[0])
                return n_freqs

    raise ValueError("Number of frequencies not found in file header")
