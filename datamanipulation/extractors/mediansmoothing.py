import numpy as np
from .extractor import Extractor


class MedianSmoothingExtractor(Extractor):
    def __init__(self, ext_obj, window_size=30, mode='same'):
        """
        Initilize the extractor.

        Args:
            ext_obj (int or Extractor): The index of the raw data column, or an Extractor object.
            window_size (positive int): The window size used for median smoothing.
            mode (str): 'full' to return the convolution at each point of overlap, 'same' returns data of the same size as the original data or 'valid' where the convolution product is only given for points where the signals overlap completely.
        """
        raw = ["time", 'x', 'y', 'pressure', 'az', 'al']

        correct_obj = (isinstance(ext_obj, int) and ext_obj in range(0, len(raw))) or isinstance(ext_obj, Extractor)

        assert correct_obj, "The input to SmoothingExtractor should be either the index of a column in the raw data (unsigned int), or an object of a class that inherits from Extractor."

        assert window_size > 0, "The window size should be > 0."

        self.ext_obj = ext_obj
        
        name_sufix = raw[ext_obj] if isinstance(ext_obj, int) else ext_obj.name
        self.name='smoothed_' + name_sufix

        self.window_size = window_size


    def _extract(self, data):
        """
        Extract the smoothed values from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with keys as features and their matrices as values.
        """
        if self.name in data.keys() and data[self.name] is not None:
            return

        if isinstance(self.ext_obj, int):
            values = data['raw'][:,self.ext_obj].copy()

        else:
            self.ext_obj._extract(data)
            values = data[self.ext_obj.name].copy().reshape(-1)

        smoothed = np.convolve(values, np.ones(self.window_size)/self.window_size, mode=self.mode)
        data[self.name] = smoothed.reshape(-1, 1)