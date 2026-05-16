from utils import data_types
import pandas as pd
import csv
import numpy as np
import os

def read(filename:str, flip=True):
    '''
    :param filename: path where the .csv is stored
    :param flip: flag to flip the matrices and arrays
    '''

    #check if the filename exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'[file_utils] Filename {filename} does not exist!')

    file_type = [".csv", ".xls", ".xlsx"]

    ext = os.path.splitext(filename)[1].lower()
    sheets = []

    if ext == ".csv":
        # process the raw data output  into a custom data structure
        raw_data = pd.read_csv(
            filename,
            delimiter=",",
            skiprows=5,
            encoding="utf-16"
        ).to_numpy()

        base = os.path.basename(filename)
        file, ext = os.path.splitext(base)
        sheets.append(file)
        raw_data_all = raw_data[:, :-1]
        n_freqs = get_n_freqs(filename)

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

            if raw_data.size==0:
                print(f"[file_utils] Sheet {sheet_names[i]} is empty.")
                continue
            sheets.append(sheet_names[i])
            raw_data = raw_data[1:, :]
            raw_data_all.append(raw_data)

        raw_data_all = np.array(raw_data_all)
        n_freqs = len(raw_data_all[0,:,0])

    else:
        raise TypeError(f'[filew_utils] Unknown file type! Curr. type = {type(file_type)}')


    data = data_types.SpectroscopyData(raw_data_all, n_freqs, ext, sheets, flip=flip)

    return data

def get_n_freqs(filepath):
    with open(filepath, 'r', encoding='utf-16') as f:
        for line in f:
            if 'freqs' in line.lower():
                # Extract number before "freqs"
                n_freqs = int(line.strip().split()[0])
                return n_freqs

    raise ValueError("Number of frequencies not found in file header")
