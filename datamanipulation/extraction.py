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
    def __init__(self, numerator_key, denominator_key, handle_inf=None, new_col_name=None):
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

    def transform(self, X, y=None):
        X_copy = X.copy()
        num_change_ext = ChangeExtractor(self.numerator_key)
        denom_change_ext = ChangeExtractor(self.denominator_key)
        num_change = num_change_ext.transform(X_copy)[num_change_ext.new_col_name]
        denom_change = denom_change_ext.transform(X_copy)[denom_change_ext.new_col_name]
        roc = num_change / denom_change
        X_copy[self.new_col_name] = roc
        if self.handle_inf:
            X_copy[self.new_col_name] = X_copy[self.new_col_name].apply(self.handle_inf)
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


feature_extraction_pipe = Pipeline([
    ('disp_x', ChangeExtractor('X', new_col_name='Displacement x')),
    ('disp_y', ChangeExtractor('Y', new_col_name='Displacement y')),
    
    ('dist_x', DistanceExtractor('X')),
    ('dist_y', DistanceExtractor('Y')),
    ('dist_xy', DistanceExtractor(('X', 'Y'))),
    
    ('ch_disp_x', ChangeExtractor('Displacement x')),
    ('ch_disp_y', ChangeExtractor('Displacement y')),
    ('ch_dist_xy', ChangeExtractor('Distance x-y')),

    ('ch_disp_x2', ChangeExtractor('Change displacement x', new_col_name="2nd change displacement x")),
    ('ch_disp_y2', ChangeExtractor('Change displacement y', new_col_name="2nd change displacement y")),
    ('ch_dist_xy2', ChangeExtractor('Change distance x-y', new_col_name="2nd change distance x-y")),

    ('ch_disp_x3', ChangeExtractor("2nd change displacement x", new_col_name="3rd change displacement x")),
    ('ch_disp_y3', ChangeExtractor("2nd change displacement y", new_col_name="3rd change displacement y")),
    ('ch_dist_xy3', ChangeExtractor("2nd change distance x-y", new_col_name="3rd change distance x-y")),

    ('ch_press', ChangeExtractor('P')),
    ('ch_al', ChangeExtractor('Al')),
    ('ch_az', ChangeExtractor('Az')),

    ('slope', ROCExtractor('Y', 'X', new_col_name='Slope')),
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
    
    print('The following features were extracted successfully:\n', data_extracted.columns[7:])
    print('Number of features:', data_extracted.columns[7:].shape[0])

    return data_extracted
