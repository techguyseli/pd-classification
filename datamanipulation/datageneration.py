from .basics import isParkinsonian
import numpy as np


def getPdImgs(info_o, data_o):
    """
    Return data as images for use when training a model.

    Args:


    Returns:
        X (list of 2-D numpy.ndarray): A multidimentional array of images of tasks' data for each participant.
        y (numpy.ndarray): A one-dimensional array of the labels of the images, 1 for parkinson, 0 for healthy control.
    """
    print('Processing started, please wait.')

    info = info_o.copy()
    data = data_o.copy()

    X = list()
    y = list()

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
            pt_image = np.array(pt_image)
            X.append(pt_image)
            y.append(parkinsonian)

    y = np.array(y)

    print('Processing done.')

    return X, y

