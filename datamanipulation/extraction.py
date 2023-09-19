import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.pipeline import Pipeline


class ChangeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, col_key, new_col_name=None):
        """
        Initilize the extractor.

        Args:
            col_key (str): The key of the column to calculate change from.
        """
        self.col_key = col_key
        self.new_col_name = ('Change ' + col_key.lower()) if new_col_name is None else new_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        change = X_copy[self.col_key].diff()
        change.iloc[0] = X_copy[self.col_key].iloc[0]
        X_copy[self.new_col_name] = change
        return X_copy


class ConvSmoothingExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, col_key, window_size=30, mode='same', new_col_name=None):
        """
        Initilize the extractor.

        Args:
            col_key (str): The key of the column to smooth.
            window_size (positive int): The window size used for median smoothing.
            mode (str): 'full' to return the convolution at each point of overlap, 'same' returns data of the same size as the original data or 'valid' where the convolution product is only given for points where the signals overlap completely.
        """
        self.col_key = col_key
        self.window_size = window_size
        self.mode = mode
        self.new_col_name = ('Smoothed ' + col_key.lower()) if new_col_name is None else new_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        values = X_copy[self.col_key]
        smoothed = np.convolve(values, np.ones(self.window_size)/self.window_size, mode=self.mode)
        X_copy[self.new_col_name] = smoothed
        return X_copy


class ROCExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, numerator_key, denominator_key, handle_inf=True, new_col_name=None):
        """
        Initilize the rate of change extractor.

        Args:
            numerator_key (str): The key of the first column, where you want to track the change in numerator_key with respect to the change in denominator_key.
            denominator_key (int): The key of the second column, where you want to track the change in numerator_key with respect to the change in denominator_key.
            handle_inf (function): A function that takes a number, and returns a new value to replace it, this is used to handle infinity values.
        """
        self.numerator_key = numerator_key
        self.denominator_key = denominator_key
        self.handle_inf = handle_inf
        self.new_col_name = ('ROC ' + numerator_key.lower() + ' / ' + denominator_key.lower()) if new_col_name is None else new_col_name

    def fit(self, X, y=None):
        return self

    def handle_inf_val(self, x, lowest_finite, highest_finite):
            if np.isneginf(x):
                return lowest_finite
            elif np.isposinf(x):
                return highest_finite
            elif np.isnan(x):
                return 0
            else:
                return x

    def transform(self, X, y=None):
        X_copy = X.copy()
        num_change_ext = ChangeExtractor(self.numerator_key)
        denom_change_ext = ChangeExtractor(self.denominator_key)
        num_change = num_change_ext.transform(X_copy)[num_change_ext.new_col_name]
        denom_change = denom_change_ext.transform(X_copy)[denom_change_ext.new_col_name]
        roc = num_change / denom_change
        if self.handle_inf:
            highest_finite = roc.sort_values().unique()[-3 if True in roc.isna().unique() else -2]
            lowest_finite = roc.sort_values().unique()[1]
            roc = roc.apply(lambda x: self.handle_inf_val(x, lowest_finite, highest_finite))
        X_copy[self.new_col_name] = roc
        return X_copy


class DistanceExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, col_keys, new_col_name=None):
        """
        Initilize the columns' keys for distance extraction.

        Args:
            col_keys (tuple or str): A tuple of the keys of x and y respectively for 2-D distance, or a single column key for 1-D distance.
        """
        self.col_keys = col_keys
        self.new_col_name = ('Distance ' + (col_keys.lower() if isinstance(col_keys, str) else (col_keys[0].lower() + '-' + col_keys[1].lower()))) if new_col_name is None else new_col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if isinstance(self.col_keys, tuple):
            x_change_extractor = ChangeExtractor(self.col_keys[0])
            y_change_extractor = ChangeExtractor(self.col_keys[1])
            change_in_x = x_change_extractor.transform(X_copy)[x_change_extractor.new_col_name]
            change_in_y = y_change_extractor.transform(X_copy)[y_change_extractor.new_col_name]
            distance = np.sqrt(change_in_x**2 + change_in_y**2)
        else:
            change_extractor = ChangeExtractor(self.col_keys)
            change = change_extractor.transform(X_copy)[change_extractor.new_col_name]
            distance = np.abs(change)
        X_copy[self.new_col_name] = distance
        return X_copy


class VelocityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, handle_inf=True, new_col_name=None):
        self.handle_inf = handle_inf
        self.new_col_name = 'Velocity x-y' if new_col_name is None else new_col_name

    def fit(self, X, y=None):
        return self

    def handle_inf_val(self, x, lowest_finite, highest_finite):
            if np.isneginf(x):
                return lowest_finite
            elif np.isposinf(x):
                return highest_finite
            elif np.isnan(x):
                return 0
            else:
                return x

    def transform(self, X, y=None):
        X_copy = X.copy()
        num_change_ext = DistanceExtractor(('X', 'Y'))
        denom_change_ext = ChangeExtractor('Time')
        num_change = num_change_ext.transform(X_copy)[num_change_ext.new_col_name]
        denom_change = denom_change_ext.transform(X_copy)[denom_change_ext.new_col_name]
        roc = num_change / denom_change
        if self.handle_inf:
            highest_finite = roc.sort_values().unique()[-3 if True in roc.isna().unique() else -2]
            lowest_finite = roc.sort_values().unique()[1]
            roc = roc.apply(lambda x: self.handle_inf_val(x, lowest_finite, highest_finite))
        X_copy[self.new_col_name] = roc
        return X_copy


class SlantExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.new_col_name = 'Slant'

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        num_change_ext = ChangeExtractor('Y')
        denom_change_ext = ChangeExtractor('X')
        num_change = num_change_ext.transform(X_copy)[num_change_ext.new_col_name]
        denom_change = denom_change_ext.transform(X_copy)[denom_change_ext.new_col_name]
        roc = np.rad2deg(np.arctan(num_change / denom_change))
        roc = roc.apply(lambda x: 0 if np.isnan(x) else x)
        X_copy[self.new_col_name] = roc
        return X_copy


feature_extraction_pipe = Pipeline([
    ('disp_x', ChangeExtractor('X', new_col_name='Displacement x')),
    ('disp_y', ChangeExtractor('Y', new_col_name='Displacement y')),
    
    ('dist_x', DistanceExtractor('X')),
    ('dist_y', DistanceExtractor('Y')),
    ('dist_xy', DistanceExtractor(('X', 'Y'))),

    ('vel_x', ROCExtractor('X', 'Time', new_col_name='Velocity x')),
    ('vel_y', ROCExtractor('Y', 'Time', new_col_name='Velocity y')),
    ('vel_xy', VelocityExtractor()),

    ('acc_x', ROCExtractor('Velocity x', 'Time', new_col_name='Acceleration x')),
    ('acc_y', ROCExtractor('Velocity y', 'Time', new_col_name='Acceleration y')),
    ('acc_xy', ROCExtractor('Velocity x-y', 'Time', new_col_name='Acceleration x-y')),

    ('jerk_x', ROCExtractor('Acceleration x', 'Time', new_col_name='Jerk x')),
    ('jerk_y', ROCExtractor('Acceleration y', 'Time', new_col_name='Jerk y')),
    ('jerk_xy', ROCExtractor('Acceleration x-y', 'Time', new_col_name='Jerk x-y')),

    ('roc_p', ROCExtractor('P', 'Time')),
    
    ('roc_al', ROCExtractor('Al', 'Time')),
    
    ('roc_az', ROCExtractor('Az', 'Time')),

    ('slope', ROCExtractor('Y', 'X', new_col_name='Slope')),
    ('slant', SlantExtractor()),
])


def extract_features(data, pipe=feature_extraction_pipe):
    """
    Extract features from data, using a pipeline that can be applied to each image in the data.

    Args:
        data (pandas.core.frame.DataFrame): The HW dataframe.
        pipe (scikit-learn Pipeline object): The pipeline to be applied to each image in 'data'.

    Returns:
        data_extracted (Pandas DataFrame): The new dataframe with extracted features.
    """
    print('Started extracting features.')
    indexes = data.index.unique()

    data_extracted = None

    for ix in indexes:
        img = data.loc[ix]

        ext_img = pipe.transform(img)

        if data_extracted is None:
            data_extracted = ext_img
            continue

        data_extracted = pd.concat([data_extracted, ext_img])
    
    print('The following features were extracted successfully:', list(data_extracted.columns[7:]))
    print('Number of features:', data_extracted.columns[7:].shape[0])

    return data_extracted
