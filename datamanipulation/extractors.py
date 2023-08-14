"""
tangetial, horizontal and vertical: acceleration, jerk, snap, crackle, pop.
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


class DurationExtractor:
    def __init__(self):
        """
        Initilize the name of the extractor.
        """
        self.name='duration'


    @classmethod
    def calculate(self, time1, time2):
        """
        Calculate the duration between 2 timestamps.

        Args:
            time1 (int): The first timestamp.
            time2 (int): The second timestamp.

        Returns:
            result (int): The duration between time1 and time2.
        """
        result = time2 - time1
        return result


    def _extract(self, data):
        """
        Extract the duration at each point from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with keys as features and their matrices as values.
        """
        if data[self.name] is not None:
            return

        time2 = data['raw'][:,0:1].copy()
        time1 = np.zeros(time2.shape)
        time1[1:,:] =  time2[:-1,:].copy()
        duration = DurationExtractor().calculate(time1, time2)
        data[self.name] = duration


class InstantaneousVelocityExtractor:
    def __init__(self, mode):
        """
        Initilize the mode for velocity extraction.

        Args:
            mode (str): In 'x' or 'y' or 'xy'.
        """
        assert mode in ['x', 'y', 'xy'], "The mode for velocity should be either 'x', 'y' or 'xy'."
        self.mode = mode
        self.name = 'velocity_' + mode


    @classmethod
    def calculate(self, displacement, duration):
        """
        Calculate the instantanious velocity at a certain point(s).

        Args:
            displacement (number or 2-D vector): The displacement traveled.
            duration (number or 2-D vector): The time it took to travel that distance.

        Returns:
            result (number or 2-D vector): The instantanious velocity when traveling 'distance' during 'time'.
        """
        result = displacement/duration
        return result


    def _extract(self, data):
        """
        Extract the instantanious velocity from the data dictionary (of one image) and add it to it.

        Args:
            data (dict): Dictionary with keys as features and their matrices as values.
        """
        if data[self.name] is not None:
            return

        duration_obj = DurationExtractor()
        if duration_obj.name in data.keys():
            duration_obj._extract(data)
            duration = data[duration_obj.name].copy()
        else:
            data[duration_obj.name] = None
            duration_obj._extract(data)
            duration = data[duration_obj.name].copy()
            data.pop(duration_obj.name)

        displacement_obj = DistanceExtractor(self.mode) if self.mode == "xy" else DisplacementExtractor(self.mode)
        if displacement_obj.name in data.keys():
            displacement_obj._extract(data)
            displacement = data[displacement_obj.name].copy()
        else:
            data[displacement_obj.name] = None
            displacement_obj._extract(data)
            displacement = data[displacement_obj.name].copy()
            data.pop(displacement_obj.name)

        velocity = np.zeros(duration.shape)
        velocity[1:,:] = InstantaneousVelocityExtractor.calculate(displacement[1:,:], duration[1:,:])
            
        data[self.name] = velocity


def extract_features(X, extractors):
    img = X
    data = {'raw' : img, DurationExtractor().name: None}

    for ext in extractors:
        data[ext.name] = None

    for ext in extractors:
        ext._extract(data)

    for key in data.copy().keys():
        if key not in [ext.name for ext in extractors]:
            data.pop(key)

    return data