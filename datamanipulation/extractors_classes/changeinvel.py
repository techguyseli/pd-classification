import numpy as np
from .instantaneousvel import InstantaneousVelocityExtractor

class ChangeInVelocityExtractor:
    def __init__(self, mode):
        """
        Initilize the mode of the extractor.

        Args:
            mode (str): In 'x' or 'y' or 'xy'.
        """
        assert mode in ['x', 'y', 'xy'], "The mode for the change in velocity should be either 'x', 'y' or 'xy'."
        self.mode = mode
        self.name = 'change_in_velocity_' + mode


    @classmethod
    def calculate(self, v1, v2):
        """
        Calculate the change in velocity between 2 records of velocity.

        Args:
            v1 (int): The first record of velocity.
            v2 (int): The second record of velocity.

        Returns:
            result (int): The change in velocity between v1 and v2.
        """
        result = v2 - v1
        return result


    def _extract(self, data):
        """
        Extract the change in velocity at each point from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with keys as features and their matrices as values.
        """
        if data[self.name] is not None:
            return

        velocity_obj = InstantaneousVelocityExtractor(self.mode)
        if velocity_obj.name in data.keys():
            velocity_obj._extract(data)
            velocity2 = data[velocity_obj.name].copy()
        else:
            data[velocity_obj.name] = None
            velocity_obj._extract(data)
            velocity2 = data[velocity_obj.name].copy()
            data.pop(velocity_obj.name)

        velocity1 = np.zeros(velocity2.shape)
        velocity1[1:,:] =  velocity2[:-1,:].copy()
        change_in_velocity = ChangeInVelocityExtractor.calculate(velocity1, velocity2)
        data[self.name] = change_in_velocity
