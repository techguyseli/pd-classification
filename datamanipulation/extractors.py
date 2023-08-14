"""
tangetial, horizontal and vertical: velocity, acceleration, jerk, snap, crackle, pop.
first derivative of pressure
slope

ROC of pressure over time
smoothed pressure
ROC of velocity over time
smoothed velocity
ROC of acceleration over time
smoothed acceleration
ROC of jerk over time
smoothed jerk
ROC of alt over time
smoothed alt
ROC of az over time
smoothed az
"""
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
            x1 (number or 2-D vector): The anterior point.
            x2 (number or 2-D vector): The posterior point, same shape as x1.

        Returns:
            result (number or 2-D vector): The displacement between x2 and x1.
        """
        result = x2 - x1
        return result


    def _extract(self, data):
        """
        Extract the displacement from the data dictionary (for one image).

        Args:
            data (dict): Dictionary provided through the caller function in datamanipulation.extractors.extract_features.

        Returns:
            data (dict): The input dictionary, but updated, this is done to optimize performance and for programming logic. 
        """
        if data[self.name] is not None:
            return data

        points = data['raw'][:, 1:3].copy()
        points = points[:,0:1] if self.mode == 'x' else points[:,1:2]

        points_i_minus1 = np.zeros(points.shape)
        points_i_minus1[1:,:] = points[0:-1, :]

        displacement = np.zeros(points.shape)
        displacement[1:,:] = DisplacementExtractor.calculate(points_i_minus1[1:,:], points[1:,:])

        data[self.name] = displacement
        return data


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
            x1 (number or 2-D vector): The anterior points in the x-axis.
            x2 (number or 2-D vector): The posterior points in the x-axis.
            y1 (number or 2-D vector): The anterior points in the y-axis.
            y2 (number or 2-D vector): The posterior points in the y-axis.

        Returns:
            result (number or 2-D vector): The Pythagorean distance between (x1, y1) and (x2, y2).
        """
        result = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return result


    @classmethod
    def calculate_1d(self, x1, x2):
        """
        Calculate the 1-D distance between 2 points or vectors.

        Args:
            x1 (number or 2-D vector): The anterior point(s).
            x2 (number or 2-D vector): The posterior point(s).

        Returns:
            result (number or 2-D vector): The distance between x2 and x1.
        """
        result = np.abs(x1 - x2)
        return result


    def _extract(self, data):
        """
        Extract the distance from the data dictionary (for one image).

        Args:
            data (dict): Dictionary provided through the caller function in datamanipulation.extractors.extract_features.

        Returns:
            data (dict): The input dictionary, but updated, this is done to optimize performance and for programming logic. 
        """
        if data[self.name] is not None:
            return data

        points = data['raw'][:, 1:3].copy()

        if self.mode == 'x':
            points = points[:,0:1]
        elif self.mode == 'y':
            points = points[:,1:2]

        points_i_minus1 = np.zeros(points.shape)
        points_i_minus1[1:,:] = points[0:-1, :]
        distance = np.zeros((points.shape[0], 1))

        if self.mode == 'xy':
            distance[1:,:] = DistanceExtractor.calculate_2d(
                points_i_minus1[1:,0:1], 
                points[1:,0:1],
                points_i_minus1[1:,1:2], 
                points[1:,1:2],
                )
        else:
            distance[1:,:] = DistanceExtractor.calculate_1d(points_i_minus1[1:,:], points[1:,:])
            
        data[self.name] = distance
        return data


class VelocityExtractor:
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
            x1 (number or 2-D vector): The anterior points in the x-axis.
            x2 (number or 2-D vector): The posterior points in the x-axis.
            y1 (number or 2-D vector): The anterior points in the y-axis.
            y2 (number or 2-D vector): The posterior points in the y-axis.

        Returns:
            result (number or 2-D vector): The Pythagorean distance between (x1, y1) and (x2, y2).
        """
        result = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return result


    @classmethod
    def calculate_1d(self, x1, x2):
        """
        Calculate the 1-D distance between 2 points or vectors.

        Args:
            x1 (number or 2-D vector): The anterior point(s).
            x2 (number or 2-D vector): The posterior point(s).

        Returns:
            result (number or 2-D vector): The distance between x2 and x1.
        """
        result = np.abs(x1 - x2)
        return result


    def _extract(self, data):
        """
        Extract the distance from the data dictionary (for one image).

        Args:
            data (dict): Dictionary provided through the caller function in datamanipulation.extractors.extract_features.

        Returns:
            data (dict): The input dictionary, but updated, this is done to optimize performance and for programming logic. 
        """
        if data[self.name] is not None:
            return data

        points = data['raw'][:, 1:3].copy()

        if self.mode == 'x':
            points = points[:,0:1]
        elif self.mode == 'y':
            points = points[:,1:2]

        points_i_minus1 = np.zeros(points.shape)
        points_i_minus1[1:,:] = points[0:-1, :]
        distance = np.zeros((points.shape[0], 1))

        if self.mode == 'xy':
            distance[1:,:] = DistanceExtractor.calculate_2d(
                points_i_minus1[1:,0:1], 
                points[1:,0:1],
                points_i_minus1[1:,1:2], 
                points[1:,1:2],
                )
        else:
            distance[1:,:] = DistanceExtractor.calculate_1d(points_i_minus1[1:,:], points[1:,:])
            
        data[self.name] = distance
        return data


def extract_features(X, extractors):
    data = {'raw' : X}

    for ext in extractors:
        data[ext.name] = None

    for ext in extractors:
        data = ext._extract(data)

    return data