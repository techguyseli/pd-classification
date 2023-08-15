import numpy as np
from .changeintime import ChangeInTimeExtractor
from .distance import DistanceExtractor
from .displacement import DisplacementExtractor

class InstantaneousVelocityExtractor:
    def __init__(self, mode):
        """
        Initilize the mode for velocity extraction.

        Args:
            mode (str): In 'x' or 'y' or 'xy'.
        """
        assert mode in ['x', 'y', 'xy'], "The mode for velocity should be either 'x', 'y' or 'xy'."
        self.mode = mode
        self.name = 'inst_velocity_' + mode


    @classmethod
    def calculate(self, displacement, change_in_time):
        """
        Calculate the instantanious velocity at a certain point(s).

        Args:
            displacement (number or (n,1) vector): The displacement traveled.
            change_in_time (number or (n,1) vector): The time it took to travel that distance.

        Returns:
            result (number or (n,1) vector): The instantanious velocity when traveling 'distance' during 'time'.
        """
        result = displacement/change_in_time
        return result


    def _extract(self, data):
        """
        Extract the instantanious velocity from the data dictionary (of one image) and add it to it.

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

        displacement_obj = DistanceExtractor(self.mode) if self.mode == "xy" else DisplacementExtractor(self.mode)
        if displacement_obj.name in data.keys():
            displacement_obj._extract(data)
            displacement = data[displacement_obj.name].copy()
        else:
            data[displacement_obj.name] = None
            displacement_obj._extract(data)
            displacement = data[displacement_obj.name].copy()
            data.pop(displacement_obj.name)

        velocity = np.zeros(change_in_time.shape)
        velocity[1:,:] = InstantaneousVelocityExtractor.calculate(displacement[1:,:], change_in_time[1:,:])
            
        data[self.name] = velocity
