import numpy as np
from .changeinacc import ChangeInAccelerationExtractor
from .changeintime import ChangeInTimeExtractor


class JerkExtractor:
    def __init__(self, mode):
        """
        Initilize the mode for jerk extraction.

        Args:
            mode (str): In 'x' or 'y' or 'xy'.
        """
        assert mode in ['x', 'y', 'xy'], "The mode for the jerk should be either 'x', 'y' or 'xy'."
        self.mode = mode
        self.name = 'jerk_' + mode


    @classmethod
    def calculate(self, change_in_acc, change_in_time):
        """
        Calculate the jerk at a certain point(s).

        Args:
            change_in_acc (number or (n,1) vector): The change in acceleration.
            change_in_time (number or (n,1) vector): The change in time.

        Returns:
            result (number or (n,1) vector): The jerk / rate of change in acceleration.
        """
        result = change_in_acc/change_in_time
        return result


    def _extract(self, data):
        """
        Extract the jerk from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with feature names as keys and their matrices as values.
        """
        if data[self.name] is not None:
            return

        change_in_time_obj = ChangeInTimeExtractor()
        if change_in_time_obj.name in data.keys():
            change_in_time_obj._extract(data)
            change_in_time = data[change_in_time_obj.name].copy()
        else:
            data[change_in_time_obj.name] = None
            change_in_time_obj._extract(data)
            change_in_time = data[change_in_time_obj.name].copy()
            data.pop(change_in_time_obj.name)

        change_in_acc_obj = ChangeInAccelerationExtractor(self.mode)
        if change_in_acc_obj.name in data.keys():
            change_in_acc_obj._extract(data)
            change_in_acc = data[change_in_acc_obj.name].copy()
        else:
            data[change_in_acc_obj.name] = None
            change_in_acc_obj._extract(data)
            change_in_acc = data[change_in_acc_obj.name].copy()
            data.pop(change_in_acc_obj.name)

        jerk = np.zeros(change_in_time.shape)
        jerk[1:,:] = JerkExtractor.calculate(change_in_acc[1:,:], change_in_time[1:,:])
            
        data[self.name] = jerk
