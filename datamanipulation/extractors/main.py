"""
tangetial, horizontal and vertical: displacement, velocity, acceleration, jerk, snap, crackle, pop.

slope / ROC y over x
ROC of pressure over time
ROC of alt over time
ROC of az over time
smoothed pressure
smoothed velocity
smoothed acceleration
smoothed jerk
smoothed alt
smoothed az
"""
from .change import ChangeExtractor
from .roc import ROCExtractor
from .distance import DistanceExtractor
from .extractor import Extractor
from .mediansmoothing import MedianSmoothingExtractor
import numpy as np


def extract_features(X, extractors):
    new_X = []
    for i in range(len(X)):
        data = {'raw' : X[i]}

        for ext in extractors:
            ext._extract(data)

        for key in data.copy().keys():
            if key not in [ext.name for ext in extractors]:
                data.pop(key)

        new_image = X[i]

        for value in data.copy().values():
            new_image = np.hstack((new_image, value))

        new_X.append(new_image)

    return new_X