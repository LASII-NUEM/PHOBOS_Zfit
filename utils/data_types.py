import numpy as np

class SpectroscopyData:
    def __init__(self, eis_data:np.ndarray, n_freqs:int=5):
        '''
        :param eis_data : raw data output from the PHOBOS acquisition system
        :param n_freqs : number of frequencies
        '''

        #check if the raw electrode data is a numpy array
        if type(eis_data) != np.ndarray:
            raise TypeError(f'[SpectroscopyData] Raw electrode data must be a numpy array! Curr. type = {type(eis_data)}')

        if type(n_freqs) != int:
            raise TypeError(f'[SpectroscopyData] Frequency number must be a integer! Curr. type = {type(n_freqs)}')

        self.n_freqs = n_freqs
        self.freq = eis_data[:,0]
        if self.freq.size != self.n_freqs:
            raise TypeError(f'[SpectroscopyData] Number of frequencies must be equal to {n_freqs}. Check your EIS file!')

        self.Z_real = eis_data[:,5]
        self.Z_imag = -eis_data[:,6]
