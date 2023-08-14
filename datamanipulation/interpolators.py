import numpy as np
import cv2

def get_data_shape(X):
    """
    Get the number of data points for each image.

    Args:
        X (list): The list of numpy arrays/images.

    Returns:
        num_points_per_img (list): A list of the number of data points for each image.
    """
    num_points_per_img = list()
    for img in X:
        num_points_per_img.append(img.shape[0])
    num_points_per_img = np.array(num_points_per_img)

    return num_points_per_img


def nearest_neighbour_interpolation(X):
    pass


def bilinear_interpolation(X):
    pass


def px_area_relation(X):
    pass


def bicubic_interpolation(X):
    pass


def lanczos_interpolation(X):
    pass