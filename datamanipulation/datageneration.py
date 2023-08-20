from .helpers import is_parkinsonian
import numpy as np
from sklearn.model_selection import train_test_split as sktts


def get_pd_hc_only(info, data, keep_label=True):
    """
    Return data of only PDs and HCs.

    Args:
        info (pandas.core.frame.DataFrame): The info dataframe.
        data (pandas.core.frame.DataFrame): The data dataframe.
        keep_label (bool, default True): Whether to keep the newly generated "PD/HC" feature in the dataframe or remove it.

    Returns:
        info (pandas.core.frame.DataFrame): The filtered info dataframe.
        data (pandas.core.frame.DataFrame): The filtered data dataframe.
    """
    label_key = 'PD/HC'
    if label_key in info.columns:
        return info, data

    info[label_key] = info.apply(is_parkinsonian, axis=1)
    info = info[info[label_key]>=0]
    data = info[[label_key]].merge(data, left_index=True, right_index=True)

    if not keep_label:
        info.drop(label_key, axis=1, inplace=True)
        data.drop(label_key, axis=1, inplace=True)

    return info, data


def stratified_train_test_split(info, data, label_key, test_size=0.3, random_state=42):
    data_participants = data[[label_key]].groupby('ID').first()

    X = data_participants.index
    y = data_participants[label_key]

    X_train, X_test, y_train, y_test = sktts(X, y, stratify=y, random_state=random_state)

    info_train = info.loc[X_train]
    info_test = info.loc[X_test]
    data_train = data.loc[X_train]
    data_test = data.loc[X_test]

    return info_train, info_test, data_train, data_test


def get_images(data, label_key):
    """
    Return data as images for use when training a model.

    Args:
        data (pandas.core.frame.DataFrame): The data dataframe.
        label_key (str): The key of the column to be used as the label.

    Returns:
        X (list of 2-D numpy.ndarray): A multidimentional array of images of tasks' data for each participant.
        y (numpy.ndarray): A one-dimensional array of the labels of the images.
    """
    X = list()
    y = list()

    exercices = data.groupby(['ID', 'Language', 'Task', label_key]).first().index

    for id_, lang, task, label in exercices:
        img = data.loc['ID'][(data.loc['ID']['Language'] == lang) & (data.loc['ID']['Task'] == task)]
        img = img[['Time', 'X', 'Y', 'P', 'Az', 'Al']]
        img.sort_values(['Time'], inplace=True)
        X.append(img.values)
        y.append(label)

    y = np.array(y)

    return X, y