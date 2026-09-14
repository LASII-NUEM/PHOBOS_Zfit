import numpy as np


class SpectroscopyData:
    def __init__(self, eis_data: np.ndarray, freq: np.ndarray, n_freqs: int = 5, file_ext: str = '.csv',
                 Type: str = 'eis', sheet_names: list = None):
        '''
        :param eis_data : raw data output from the PHOBOS acquisition system
        :param n_freqs : number of frequencies
        :param flip: flag to flip the matrices and arrays
        '''

        # check if the raw electrode data is a numpy array
        if type(eis_data) != np.ndarray:
            raise TypeError(
                f'[SpectroscopyData] Raw electrode data must be a numpy array! Curr. type = {type(eis_data)}')

        if type(freq) != np.ndarray:
            raise TypeError(
                f'[SpectroscopyData] frequency data must be a numpy array! Curr. type = {type(freq)}')

        if type(n_freqs) != int:
            raise TypeError(f'[SpectroscopyData] Frequency number must be a integer! Curr. type = {type(n_freqs)}')

        if type(sheet_names) != list:
            raise TypeError(f'[SpectroscopyData] Sheet names must be a string list! Curr. type = {type(sheet_names)}')

        ext = [".csv", ".xls", ".xlsx"]
        if file_ext not in ext:
            raise TypeError(f'[file_utils] Unknown file type! Curr. type = {ext}')

        file_type = ["eis", "phobos"]
        Type = Type.lower()
        if Type not in file_type:
            raise TypeError(f'[file_utils] Unknown file type! Curr. type = {Type}')

        self.freq = freq
        self.n_freqs = n_freqs

        if self.freq.size != self.n_freqs:
            raise TypeError(
                f'[SpectroscopyData] Number of frequencies must be equal to {n_freqs}. Check your EIS file!')

        self.sheet_names = sheet_names

        if Type == "eis":
            if file_ext == ".csv":

                self.Z_real = eis_data[:, 5]
                self.Z_imag = -eis_data[:, 6]

            elif file_ext in [".xls", ".xlsx"]:
                self.freq = np.flip(self.freq)
                self.Z_real = np.flip(eis_data[:,:,2], axis=1)
                self.Z_imag = np.flip(eis_data[:,:,3], axis=1)
                self.sheets = eis_data.shape[0]

        elif Type == "phobos":

            # organize the data based on the CSV format
            valid_electrodes = eis_data[:, 2:]  # filter the array from the first electrode reading
            # process capacitance and resistance separately
            idx_cp = np.arange(0, int(2 * len(self.freq)), 2)  # indexes of each capacitance reading
            self.Cp = valid_electrodes[:, idx_cp]  # update capacitance readings
            idx_rp = np.arange(1, int(2 * len(self.freq)), 2)  # indexes of each resistance reading
            self.Rp = valid_electrodes[:, idx_rp]  # update resistance readings

            if np.all(np.char.strip(eis_data[:, 1].astype(str)) == ""):
                self.cell = "Commercial-cell"
                self.Cp_avg = np.mean(self.Cp, axis=0)
                self.Rp_avg = np.mean(self.Rp, axis=0)
                self.Z_real, self.Z_imag = self.nyquist()

            else:
                self.cell = "Muiltielectrode-cell"
                self.timestamp = eis_data[:, 0]
                raw_modes = list(dict.fromkeys(eis_data[:, 1]))
                self.modes = np.array([str(m).strip().replace('d:', '') for m in raw_modes], dtype=object)
                self.n_modes = len(self.modes)  # length off modes

                n_complete_loops = 0
                for i in range(0, len(eis_data), self.n_modes):
                    loop = eis_data[i:i + self.n_modes, :]
                    if len(loop) < self.n_modes:
                        print(f"[PHOBOS] Incomplete last loop discarded "
                              f"({len(loop)}/{self.n_modes} modes).")
                        break

                    n_complete_loops += 1
                if n_complete_loops == 0:
                    raise ValueError("[PHOBOS] No complete electrode measurement loop found.")

                n_valid_rows = n_complete_loops * self.n_modes
                eis_data = eis_data[:n_valid_rows, :]

                self.n_samples = len(eis_data[:, 0]) // self.n_modes
                self.Cp = np.reshape(self.Cp[:n_valid_rows,:], [self.n_samples, self.n_modes, len(self.freq)]).transpose(0, 2, 1)
                self.Rp = np.reshape(self.Rp[:n_valid_rows,:], [self.n_samples, self.n_modes, len(self.freq)]).transpose(0, 2, 1)
                self.Cp_avg = np.mean(self.Cp, axis=0)
                self.Rp_avg = np.mean(self.Rp, axis=0)

                self.Z_real, self.Z_imag = self.nyquist()

        else:
            raise TypeError(f'[file_utils] Unknown file type! Curr. type = {Type}')

    def nyquist(self):

        self.omegas = 2 * np.pi * self.freq

        if self.cell == "Commercial-cell":

            self.Z_real = self.Cp_avg / (1 + (self.omegas[:] * self.Cp_avg * self.Rp_avg) ** 2)
            self.Z_imag = (self.omegas[:] * self.Cp_avg * (self.Rp_avg ** 2)) / (
                    1 + (self.omegas[:] * self.Cp_avg * self.Rp_avg) ** 2)

        elif self.cell == "Muiltielectrode-cell":

            self.Z_real = self.Cp_avg / (1 + (self.omegas[:, np.newaxis] * self.Cp_avg * self.Rp_avg) ** 2)
            self.Z_imag = (self.omegas[:, np.newaxis] * self.Cp_avg * (self.Rp_avg ** 2)) / (
                    1 + (self.omegas[:, np.newaxis] * self.Cp_avg * self.Rp_avg) ** 2)

        else:
            raise TypeError(f'[file_utils] Unknown cell type! Curr. type = {self.cell}')

        return self.Z_real, -self.Z_imag
