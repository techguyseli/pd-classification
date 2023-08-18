import numpy as np
from .extractor import Extractor


class DistanceExtractor(Extractor):
    def __init__(self, mode):
        """
        Initilize the mode for distance extraction.

        Args:
            mode (str): In 'x' or 'y' or 'xy'.
        """
        assert mode in ['x', 'y', 'xy'], "The mode for distance should be either 'x', 'y' or 'xy'."
        self.mode = mode
        self.name = 'distance_' + mode


    def _extract(self, data):
        """
        Extract the distance from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with keys as features and their matrices as values.
        """
        if self.name in data.keys() and data[self.name] is not None:
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
            x1 = points_i_minus1[1:,0:1]
            x2 = points[1:,0:1]
            y1 = points_i_minus1[1:,1:2]
            y2 = points[1:,1:2]
            distance[1:,:] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        else:
            v1 = points_i_minus1[1:,:]
            v2 = points[1:,:]
            distance[1:,:] = np.abs(v1 - v2)
            
        data[self.name] = distance