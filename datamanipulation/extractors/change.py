import numpy as np
from .extractor import Extractor


class ChangeExtractor(Extractor):
    def __init__(self, ext_obj):
        """
        Initilize the extractor.

        Args:
            ext_obj (int or Extractor): The index of the raw data column, or an Extractor object.
        """
        raw = ["time", 'x', 'y', 'pressure', 'az', 'al']

        correct_obj = (isinstance(ext_obj, int) and ext_obj in range(0, len(raw))) or isinstance(ext_obj, Extractor)

        assert correct_obj, "The input to ChangeExtractor should be either the index of a column in the raw data (unsigned int), or an object of a class that inherits from Extractor."

        self.ext_obj = ext_obj
        
        name_sufix = raw[ext_obj] if isinstance(ext_obj, int) else ext_obj.name
        self.name='change_in_' + name_sufix


    def _extract(self, data):
        """
        Extract the change in values at each point from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with keys as features and their matrices as values.
        """
        if self.name in data.keys() and data[self.name] is not None:
            return

        if isinstance(self.ext_obj, int):
            values2 = data['raw'][:,self.ext_obj].reshape((-1, 1)).copy()

        else:
            self.ext_obj._extract(data)
            values2 = data[self.ext_obj.name].copy()

        values1 = np.zeros(values2.shape)
        values1[1:,:] =  values2[:-1,:].copy()
        change_in_values = np.zeros(values2.shape)
        change_in_values[1:,:] = values2[1:,:] - values1[1:,:]
        data[self.name] = change_in_values