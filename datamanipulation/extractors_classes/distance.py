import numpy as np


class DistanceExtractor:
    def __init__(self, mode):
        """
        Initilize the mode for distance extraction.

        Args:
            mode (str): In 'x' or 'y' or 'xy'.
        """
        assert mode in ['x', 'y', 'xy'], "The mode for distance should be either 'x', 'y' or 'xy'."
        self.mode = mode
        self.name = 'distance_' + mode


    @classmethod
    def calculate_2d(self, x1, x2, y1, y2):
        """
        Calculate the Pythagorean distance between 2 points or vectors.

        Args:
            x1 (number or (n,1) vector): The anterior points in the x-axis.
            x2 (number or (n,1) vector): The posterior points in the x-axis.
            y1 (number or (n,1) vector): The anterior points in the y-axis.
            y2 (number or (n,1) vector): The posterior points in the y-axis.

        Returns:
            result (number or (n,1) vector): The Pythagorean distance between (x1, y1) and (x2, y2).
        """
        result = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return result


    @classmethod
    def calculate_1d(self, x1, x2):
        """
        Calculate the 1-D distance between 2 points or vectors.

        Args:
            x1 (number or (n,1) vector): The anterior point(s).
            x2 (number or (n,1) vector): The posterior point(s).

        Returns:
            result (number or (n,1) vector): The distance between x2 and x1.
        """
        result = np.abs(x1 - x2)
        return result


    def _extract(self, data):
        """
        Extract the distance from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with keys as features and their matrices as values.
        """
        if data[self.name] is not None:
            return

        points = data['raw'][:, 1:3].copy()

        if self.mode == 'x':
            points = points[:,0:1]
        elif self.mode == 'y':
            points = points[:,1:2]

        points_i_minus1 = np.zeros(points.shape)
        points_i_minus1[1:,:] = points[0:-1, :].copy()

        distance = np.zeros((points.shape[0], 1))
        if self.mode == 'xy':
            distance[1:,:] = DistanceExtractor.calculate_2d(
                points_i_minus1[1:,0:1], 
                points[1:,0:1],
                points_i_minus1[1:,1:2], 
                points[1:,1:2]
                )
        else:
            distance[1:,:] = DistanceExtractor.calculate_1d(points_i_minus1[1:,:], points[1:,:])
            
        data[self.name] = distance
