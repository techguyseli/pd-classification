from .helpers import is_parkinsonian
import numpy as np
from sklearn.model_selection import train_test_split as sktts
from dataaccess.filedatareader import FileDataReader


def get_pd_hc_only(info, data):
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
    label_key = 'PD'
    if label_key in info.columns:
        return info, data

    info[label_key] = info.apply(is_parkinsonian, axis=1)
    info = info[info[label_key]>=0]
    data = data.reset_index(['Language', 'Task'])
    data = info[[label_key]].merge(data, left_on='ID', right_on='ID')
    
    data.reset_index('ID', inplace=True)
    data = FileDataReader('.')._postprocess_tasks_dataframe(data)

    return info, data


def stratified_train_test_split(info, data, label_key, test_size=0.3, random_state=42):
    df = data.groupby(['ID', 'Language', 'Task']).first()
    df.reset_index(['Language', 'Task'], inplace=True)
    df = df.merge(info, on='ID').reset_index('ID').set_index(['ID', 'Language', 'Task'])

    label_key += '_x'

    X = df.index
    y = df.reset_index()[[label_key, 'Gender']]

    X_train, X_test, y_train, y_test = sktts(X, y, stratify=y, random_state=random_state)

    info_X_train = np.apply_along_axis(lambda x: list(x), 0, X_train)[:,0]
    info_X_test = np.apply_along_axis(lambda x: list(x), 0, X_test)[:,0]

    info_train = info.loc[info_X_train]
    info_test = info.loc[info_X_test]
    data_train = data.loc[X_train].sort_index()
    data_test = data.loc[X_test].sort_index()

    return info_train, info_test, data_train, data_test


def match_age_gender_pd(info, data):
    """
    Match PDs and HCs in age and gender.

    Args:
        info (pandas.DataFrame): The info dataframe.
        data (pandas.DataFrame): The data dataframe.
        
    Returns:
        info, data (pandas.DataFrame): An age and gender matched info and data dataframes.
    """
    df = info.merge(data.groupby('ID').first(), left_on='ID', right_on='ID')
    min_age, max_age = df[df['PD_x']==1]['Age'].sort_values()[[0, -1]]
    num_females, num_males = df[(df['PD_x']==1) & (df['Age']>=min_age) & (df['Age']<=max_age)].groupby('Gender').count()['Age'].sort_values()
    age_matched_hcs = df[(df['PD_x']==0) & (df['Age']>=min_age) & (df['Age']<=max_age)]
    f_hcs = age_matched_hcs[age_matched_hcs['Gender']=='Female']
    m_hcs = age_matched_hcs[age_matched_hcs['Gender']=='Male']

    import numpy as np
    np.random.seed(seed=42)
    f_ixs = np.random.choice(f_hcs.shape[0], num_females, replace=False)
    m_ixs = np.random.choice(m_hcs.shape[0], num_males, replace=False)

    to_keep = list(df[df['PD_x']==1].index) + list(f_hcs.iloc[f_ixs].index) + list(m_hcs.iloc[m_ixs].index)
    
    return info.loc[to_keep], data.loc[to_keep]


def get_samples(data, label_key):
    X = list()
    y = list()
    cols = list(data.columns)
    cols.remove(label_key)
    
    for ix in data.index.unique():
        X.append(data.loc[ix][cols].values)
        y.append(data.loc[ix].iloc[0][label_key])

    y = np.array(y)

    return X, y