import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ChangeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, col_ix):
        """
        Initilize the extractor.

        Args:
            col_ix (int): The index of the column to calculate change from.
        """
        self.col_ix = col_ix

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        change = np.zeros(X.shape[0])
        change[0] = X[0, self.col_ix]
        change[1:] = np.diff(X[:, self.col_ix])
        return np.c_[X, change]


class DistanceExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, ix):
        """
        Initilize the indices for distance extraction.

        Args:
            ix (tuple or int): A tuple of the indexes of x and y, respectively, for 2-D distance, or the index a column for 1-D distance.
        """
        assert isinstance(ix, int) or isinstance(ix, tuple), "ix should be either a tuple of the indexes of x and y, respectively, for 2-D distance, or the index a column for 1-D distance."
        self.ix = ix

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(self.ix, tuple):
            change_in_x = ChangeExtractor(self.ix[0]).transform(X)[:, -1]
            change_in_y = ChangeExtractor(self.ix[1]).transform(X)[:, -1]
            distance = np.sqrt((change_in_x)**2 + (change_in_y)**2)
        else:
            change_in_vals = ChangeExtractor(self.ix).transform(X)[:, -1]
            distance = np.abs(change_in_vals)
        return np.c_[X, distance]


class MedianSmoothingExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, ix, window_size=30, mode='same'):
        """
        Initilize the extractor.

        Args:
            ix (int): The index of the column to smooth.
            window_size (positive int): The window size used for median smoothing.
            mode (str): 'full' to return the convolution at each point of overlap, 'same' returns data of the same size as the original data or 'valid' where the convolution product is only given for points where the signals overlap completely.
        """
        self.ix = ix
        self.window_size = window_size
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        values = X[:, self.ix]
        smoothed = np.convolve(values, np.ones(self.window_size)/self.window_size, mode=self.mode)
        return np.c_[X, smoothed]


class ROCExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, ix1, ix2, handle_inf=None):
        """
        Initilize the rate of change extractor.

        Args:
            ix1 (int): The index of the first column, where you want to track the change in ix1 with respect to the change in ix2.
            ix2 (int): The index of the second column, where you want to track the change in ix1 with respect to the change in ix2.
            handle_inf (function): A function that takes a number, and returns a new value to replace it, this is used to handle infinity values in numpy.
        """
        self.ix1 = ix1
        self.ix2 = ix2
        self.handle_inf = handle_inf

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        var1_change = ChangeExtractor(self.ix1).transform(X)[:, -1]
        var2_change = ChangeExtractor(self.ix2).transform(X)[:, -1]
        roc = var1_change / var2_change
        return np.c_[X, roc]


