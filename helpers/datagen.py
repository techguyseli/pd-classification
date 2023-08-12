import numpy as np
from .basics import isParkinsonian


def getPdImgs(info_o, data_o, scaling_method=None):
    """
    Return data as images for use when training a model.

    Args:


    Returns:
        X (numpy.ndarray): A multidimentional array of images of tasks' data for each participant.
        y (numpy.ndarray): A one-dimensional array of the labels of the images, 1 for parkinson, 0 for healthy control.
    """
    print('Processing started, please wait.')

    info = info_o.copy()
    data = data_o.copy()

    X_l = list()
    y_l = list()

    participants = data.index.unique()

    for p in participants:
        p_info = info.loc[p]
        parkinsonian = isParkinsonian(p_info)
        if parkinsonian == -1:
            continue

        p_data = data.loc[p]
        p_tasks = p_data.Task.unique()

        for t in p_tasks:
            pt_image = p_data[p_data.Task==t]
            pt_image = pt_image[['Time', 'X', 'Y', 'P', 'Az', 'Al']]
            pt_image.sort_values(['Time'], inplace=True)
            pt_image = pt_image.to_numpy() if scaling_method is None else scaling_method(pt_image.to_numpy())
            X_l.append(pt_image)
            y_l.append(parkinsonian)


    y = np.array(y_l)

    X = X_l if scaling_method is None else np.array(X_l)

    print('Processing done.')

    return X, y

