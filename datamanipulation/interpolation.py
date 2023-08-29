from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
import numpy as np


class Interpolator(BaseEstimator, TransformerMixin):
    def __init__(self, label_col, resize_method=tf.image.ResizeMethod.BILINEAR, new_shape_mode="mean"):
        """
        Initialize the interpolator.

        Args:
            label_col (str): The column to be taken as label to the supervised data.
            resize_method (tensorflow.image.ResizeMethod.*, default: tensorflow.images.ResizeMethod.BILINEAR): A tensorflow image resizing method.
            new_shape_mode (str): One of the following:
                'min': For the new shape to be the minimum of the number of datapoints.
                'max': For the new shape to be the maximum of the number of datapoints.
                'mean': For the new shape to be the mean of the number of datapoints.
                '25%': For the new shape to be the first quartile of the number of datapoints.
                '50%': For the new shape to be the median of the number of datapoints.
                '75%': For the new shape to be the third quartile of the number of datapoints.
        """
        assert new_shape_mode in ['min', 'max', 'mean', '25%', '50%', '75%'], "The new_shape_mode should be one of the following: 'min', 'max', 'mean', '25%', '50%' or '75%'."

        self.resize_method = resize_method
        self.new_shape_mode = new_shape_mode
        self.label_col = label_col


    def fit(self, X, y=None):
        """
        Find the new shape.
        """
        self.new_length_ = int(X.groupby(['ID', 'Language', 'Task']).count()['X'].describe()[self.new_shape_mode])
        return self


    def transform(self, X, y=None):
        """
        Scale the images.
        """
        print('Started image interpolation.')

        grouping = X.groupby(['ID', 'Language', 'Task']).first()

        new_y = grouping[self.label_col].values

        indexes = grouping.index

        images = [X.loc[ix].drop(self.label_col, axis=1).values for ix in indexes]

        new_shape = (self.new_length_, images[0].shape[1])
        
        for i in range(len(images)):
            images[i] = tf.image.resize(
                images[i].reshape((images[i].shape[0], images[i].shape[1], 1)),
                new_shape,
                method=self.resize_method
            ).numpy().reshape(new_shape)

        new_X = np.array(images)

        print('Interpolation done.')

        return new_X, new_y