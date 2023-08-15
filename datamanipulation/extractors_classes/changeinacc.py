import numpy as np
from .acceleration import AccelerationExtractor

class ChangeInAccelerationExtractor:
    def __init__(self, mode):
        """
        Initilize the mode of the extractor.

        Args:
            mode (str): In 'x' or 'y' or 'xy'.
        """
        assert mode in ['x', 'y', 'xy'], "The mode for the change in acceleration should be either 'x', 'y' or 'xy'."
        self.mode = mode
        self.name = 'change_in_acceleration_' + mode


    @classmethod
    def calculate(self, a1, a2):
        """
        Calculate the change in acceleration between 2 points.

        Args:
            a1 (int): The first point of acceleration.
            a2 (int): The second point of acceleration.

        Returns:
            result (int): The change in acceleration between a1 and a2.
        """
        result = a2 - a1
        return result


    def _extract(self, data):
        """
        Extract the change in acceleration at each point from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with keys as features and their matrices as values.
        """
        if data[self.name] is not None:
            return

        acc_obj = AccelerationExtractor(self.mode)
        if acc_obj.name in data.keys():
            acc_obj._extract(data)
            acc2 = data[acc_obj.name].copy()
        else:
            data[acc_obj.name] = None
            acc_obj._extract(data)
            acc2 = data[acc_obj.name].copy()
            data.pop(acc_obj.name)

        acc1 = np.zeros(acc2.shape)
        acc1[1:,:] =  acc2[:-1,:].copy()
        change_in_acc = ChangeInAccelerationExtractor.calculate(acc1, acc2)
        data[self.name] = change_in_acc
