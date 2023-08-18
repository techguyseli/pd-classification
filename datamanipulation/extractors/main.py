"""
tangetial, horizontal and vertical: snap, crackle, pop.


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


def extract_features(X, extractors):
    img = X
    data = {'raw' : img}

    for ext in extractors:
        ext._extract(data)

    for key in data.copy().keys():
        if key not in [ext.name for ext in extractors]:
            data.pop(key)

    return data