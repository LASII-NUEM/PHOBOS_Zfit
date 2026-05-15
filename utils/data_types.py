import numpy as np

class SpectroscopyData:
    def __init__(self, eis_data:np.ndarray, n_freqs:int=5, file_type:str='.csv', sheet_names:list=None, flip=False):
        '''
        :param eis_data : raw data output from the PHOBOS acquisition system
        :param n_freqs : number of frequencies
        :param flip: flag to flip the matrices and arrays
        '''

        #check if the raw electrode data is a numpy array
        if type(eis_data) != np.ndarray:
            raise TypeError(f'[SpectroscopyData] Raw electrode data must be a numpy array! Curr. type = {type(eis_data)}')

        if type(n_freqs) != int:
            raise TypeError(f'[SpectroscopyData] Frequency number must be a integer! Curr. type = {type(n_freqs)}')

        if type(sheet_names) != list:
            raise TypeError(f'[SpectroscopyData] Sheet names must be a string list! Curr. type = {type(sheet_names)}')

        ext = [".csv", ".xls", ".xlsx"]
        if file_type not in ext:
            raise TypeError(f'[file_utils] Unknown file type! Curr. type = {type(ext)}')

        self.n_freqs = n_freqs
        self.sheet_names = sheet_names

        if file_type == ".csv":
            self.freq = eis_data[:, 0]
            if self.freq.size != self.n_freqs:
                raise TypeError(
                    f'[SpectroscopyData] Number of frequencies must be equal to {n_freqs}. Check your EIS file!')
            self.Z_real = eis_data[:, 5]
            self.Z_imag = -eis_data[:, 6]
        elif file_type in [".xls", ".xlsx"]:
            self.freq = eis_data[0,:,0]
            if flip:
                self.freq = self.freq[::-1]
            if self.freq.size != self.n_freqs:
                raise TypeError(
                    f'[SpectroscopyData] Number of frequencies must be equal to {n_freqs}. Check your EIS file!')
            self.Z_real = eis_data[:,:, 2]
            self.Z_imag = -eis_data[:,:, 3]
            if flip:
                self.Z_real = np.flip(self.Z_real, axis=1)
                self.Z_imag = np.flip(-self.Z_imag, axis=1)
            self.sheets = eis_data.shape[0]

