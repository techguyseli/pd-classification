import numpy as np

class DisplacementExtractor:
    def __init__(self, mode):
        """
        Initilize the mode for displacement extraction.

        Args:
            mode (str): In 'x' or 'y'.
        """
        assert mode in ['x', 'y'], "The mode for displacement should be either 'x' or 'y'."
        self.mode = mode
        self.name = 'displacement_' + mode


    @classmethod
    def calculate(self, x1, x2):
        """
        Calculate the displacement between 2 points or vectors.

        Args:
            x1 (number or (n,1) vector): The anterior point.
            x2 (number or (n,1) vector): The posterior point, same shape as x1.

        Returns:
            result (number or (n,1) vector): The displacement between x2 and x1.
        """
        result = x2 - x1
        return result


    def _extract(self, data):
        """
        Extract the displacement from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with keys as features and their matrices as values.
        """
        if data[self.name] is not None:
            return

        points = data['raw'][:, 1:3].copy()
        points = points[:,0:1] if self.mode == 'x' else points[:,1:2]

        points_i_minus1 = np.zeros(points.shape)
        points_i_minus1[1:,:] = points[0:-1, :].copy()

        displacement = np.zeros(points.shape)
        displacement[1:,:] = DisplacementExtractor.calculate(points_i_minus1[1:,:], points[1:,:])

        data[self.name] = displacement
