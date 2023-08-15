"""
tangetial, horizontal and vertical: jerk, snap, crackle, pop.

slope

ROC of pressure over time
smoothed pressure
smoothed velocity
smoothed acceleration
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
            displacement (number or 2-D vector): The displacement traveled.
            change_in_time (number or 2-D vector): The time it took to travel that distance.

        Returns:
            result (number or 2-D vector): The instantanious velocity when traveling 'distance' during 'time'.
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


class ChangeInVelocityExtractor:
    def __init__(self, mode):
        """
        Initilize the name of the extractor.

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
        Extract the change in time at each point from the data dictionary (of one image) and add it to it.

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
            change_in_velocity (number or 2-D vector): The change in velocity.
            change_in_time (number or 2-D vector): The change in time.

        Returns:
            result (number or 2-D vector): The acceleration / rate of change of velocity.
        """
        result = change_in_velocity/change_in_time
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

        change_in_vel_obj = InstantaneousVelocityExtractor(self.mode)
        if change_in_vel_obj.name in data.keys():
            change_in_vel_obj._extract(data)
            change_in_vel = data[change_in_vel_obj.name].copy()
        else:
            data[change_in_vel_obj.name] = None
            change_in_vel_obj._extract(data)
            change_in_vel = data[change_in_vel_obj.name].copy()
            data.pop(change_in_vel_obj.name)

        acceleration = np.zeros(change_in_time.shape)
        acceleration[1:,:] = InstantaneousVelocityExtractor.calculate(change_in_vel[1:,:], change_in_time[1:,:])
            
        data[self.name] = acceleration


def extract_features(X, extractors):
    img = X
    data = {'raw' : img, ChangeInTimeExtractor().name: None}

    for ext in extractors:
        data[ext.name] = None

    for ext in extractors:
        ext._extract(data)

    for key in data.copy().keys():
        if key not in [ext.name for ext in extractors]:
            data.pop(key)

    return data