"""
tangetial, horizontal and vertical: snap, crackle, pop.

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
from .extractors_classes.displacement import DisplacementExtractor
from .extractors_classes.distance import DistanceExtractor
from .extractors_classes.changeintime import ChangeInTimeExtractor
from .extractors_classes.instantaneousvel import InstantaneousVelocityExtractor
from .extractors_classes.changeinvel import ChangeInVelocityExtractor
from .extractors_classes.acceleration import AccelerationExtractor
from .extractors_classes.changeinacc import ChangeInAccelerationExtractor
from .extractors_classes.jerk import JerkExtractor
from .extractors_classes.slope import SlopeExtractor


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