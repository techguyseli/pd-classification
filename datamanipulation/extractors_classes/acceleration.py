import numpy as np
from .changeintime import ChangeInTimeExtractor
from .changeinvel import ChangeInVelocityExtractor

class AccelerationExtractor:
    def __init__(self, mode):
        """
        Initilize the mode for acceleration extraction.

        Args:
            mode (str): In 'x' or 'y' or 'xy'.
        """
        assert mode in ['x', 'y', 'xy'], "The mode for acceleration should be either 'x', 'y' or 'xy'."
        self.mode = mode
        self.name = 'acceleration_' + mode


    @classmethod
    def calculate(self, change_in_velocity, change_in_time):
        """
        Calculate the acceleration at a certain point(s).

        Args:
            change_in_velocity (number or (n,1) vector): The change in velocity.
            change_in_time (number or (n,1) vector): The change in time.

        Returns:
            result (number or (n,1) vector): The acceleration / rate of change of velocity.
        """
        result = change_in_velocity / change_in_time
        return result


    def _extract(self, data):
        """
        Extract the acceleration from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with keys as features and their matrices as values.
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

        change_in_vel_obj = ChangeInVelocityExtractor(self.mode)
        if change_in_vel_obj.name in data.keys():
            change_in_vel_obj._extract(data)
            change_in_vel = data[change_in_vel_obj.name].copy()
        else:
            data[change_in_vel_obj.name] = None
            change_in_vel_obj._extract(data)
            change_in_vel = data[change_in_vel_obj.name].copy()
            data.pop(change_in_vel_obj.name)

        acceleration = np.zeros(change_in_time.shape)
        acceleration[1:,:] = AccelerationExtractor.calculate(change_in_vel[1:,:], change_in_time[1:,:])
            
        data[self.name] = acceleration
