import numpy as np
import tensorflow as tf

def scale_imgs(X, scale_method, interpolation_method):
    """
    Scale a list of images.

    Args:
        X (list): The list of 2-D images.
        scale_method (list): The measure that the new shape should take, 'min', 'max', 'avg'
    """
    for i in range(len(X)):
        new_shape = (original_img.shape[0] - 300, original_img.shape[1])

scaled_img = tf.image.resize(
    original_img.reshape((original_height, original_width, 1)),
    new_shape,
    method=tf.image.ResizeMethod.BILINEAR
)
scaled_img = tf.reshape(scaled_img, new_shape)

    return X
