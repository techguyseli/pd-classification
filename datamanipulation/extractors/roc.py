import numpy as np
from .extractor import Extractor
from .change import ChangeExtractor


class ROCExtractor(Extractor):
    def _correct_inputs(self, objs, raw_data_len):
        for obj in objs:
            correct_obj = (isinstance(obj, int) and obj in range(0, raw_data_len)) or isinstance(obj, Extractor)

            if not correct_obj:
                return False

        return True


    def __init__(self, var1, var2):
        """
        Initilize the rate of change extractor.

        Args:
            var1 (int or Extractor): The index of the raw data column, or an Extractor object, where you want to track the change in var1 with respect to the change in var2.
            var2 (int or Extractor): The index of the raw data column, or an Extractor object, where you want to track the change in var1 with respect to the change in var2.
        """
        raw = ["time", 'x', 'y', 'pressure', 'az', 'al']

        assert self._correct_inputs((var1, var2), len(raw)), "The inputs to ROCExtractor should be either the index of a column in the raw data (unsigned int), or an object of a class that inherits from Extractor."

        self.var1 = var1
        self.var2 = var2

        var1_name_sufix = raw[var1] if isinstance(var1, int) else var1.name
        var2_name_sufix = raw[var2] if isinstance(var2, int) else var2.name

        self.name='roc_' + '_'.join((var1_name_sufix, var2_name_sufix))


    def _extract(self, data):
        """
        Extract the roc of var1 in var2 from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with feature names as keys and their matrices as values.
        """
        if self.name in data.keys() and data[self.name] is not None:
            return

        ch_var1_obj = ChangeExtractor(self.var1)
        ch_var2_obj = ChangeExtractor(self.var2)

        ch_var1_obj._extract(data)
        ch_var2_obj._extract(data)

        var1_change = data[ch_var1_obj.name].copy()
        var2_change = data[ch_var2_obj.name].copy()

        roc = np.zeros(var1_change.shape)
        roc[1:,:] = var1_change[1:,:] / var2_change[1:,:]
            
        data[self.name] = roc

