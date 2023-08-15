import numpy as np

class SlopeExtractor:
    def __init__(self):
        """
        Initilize the slope extractor.
        """
        self.name = 'slope'


    @classmethod
    def calculate(self, x1, x2, y1, y2):
        """
        Calculate the displacement between 2 points or vectors.

        Args:
            x1 (number or (n,1) vector): The anterior point in x-axis.
            x2 (number or (n,1) vector): The posterior point in x-axis.
            y1 (number or (n,1) vector): The anterior point in y-axis.
            y2 (number or (n,1) vector): The posterior point in y-axis.

        Returns:
            result (number or (n,1) vector): The slope of the 2 points.
        """
        result = (y2 - y1) / (x2 - x1)
        return result


    def _extract(self, data):
        """
        Extract the slope from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with keys as features and their matrices as values.
        """
        if data[self.name] is not None:
            return

        points = data['raw'][:, 1:3].copy()
        y2 = points[:,1:2]
        x2 = points[:,0:1]

        y1 = np.zeros(y2.shape)
        y1[1:,:] = y2[0:-1, :]

        x1 = np.zeros(x2.shape)
        x1[1:,:] = x2[0:-1, :]

        slope = np.zeros(x1.shape)
        slope[1:,:] = SlopeExtractor.calculate(x1[1:,:], x2[1:,:], y1[1:,:], y2[1:,:])

        data[self.name] = slope

