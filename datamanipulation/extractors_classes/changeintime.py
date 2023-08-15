import numpy as np


class ChangeInTimeExtractor:
    def __init__(self):
        """
        Initilize the name of the extractor.
        """
        self.name='change_in_time'


    @classmethod
    def calculate(self, time1, time2):
        """
        Calculate the change in time between 2 timestamps.

        Args:
            time1 (int): The first timestamp.
            time2 (int): The second timestamp.

        Returns:
            result (int): The change in time between time1 and time2.
        """
        result = time2 - time1
        return result


    def _extract(self, data):
        """
        Extract the change in time at each point from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with keys as features and their matrices as values.
        """
        if data[self.name] is not None:
            return

        time2 = data['raw'][:,0:1].copy()
        time1 = np.zeros(time2.shape)
        time1[1:,:] =  time2[:-1,:].copy()
        change_in_time = ChangeInTimeExtractor().calculate(time1, time2)
        data[self.name] = change_in_time
