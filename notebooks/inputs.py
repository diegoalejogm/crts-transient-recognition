import pandas as pd
import numpy as np
import inspect
from sklearn.model_selection import train_test_split


DATA_PATH = '../data/'
FEATURES_PATH = DATA_PATH + 'features/'

def top_transient_classes():
    return ['SN', 'CV', 'AGN', 'HPM', 'Blazar', 'Flare']

def load_transient_features(min_obs, num_features, oversample):
    indir = FEATURES_PATH
    oversample_path = '_os' if oversample else ''
    filename_format = 'T_{}obs_{}feat{}.pickle'.format(min_obs, num_features, oversample_path)
    filename = filename_format.format(min_obs, num_features)  
    inpath = indir + filename
    df_feat_tran = pd.read_pickle(inpath)
    if oversample:
        task_name = inspect.stack()[1][3]
        task = globals()[task_name]
        df_feat_tran = __filter_oversampled_transient__(df_feat_tran, task, min_obs, num_features)
    return df_feat_tran

def load_nontransient_features(min_obs, num_features, oversample):
    indir = FEATURES_PATH
    oversample_path = '_os' if oversample else ''
    filename_format = 'NT_{}obs_{}feat{}.pickle'.format(min_obs, num_features, oversample_path)
    filename = filename_format.format(min_obs, num_features)  
    inpath = indir + filename
    df_feat_nont = pd.read_pickle(inpath)
    if oversample:
        task_name = inspect.stack()[1][3]
        task = globals()[task_name]
        df_feat_nont = __filter_oversampled_nontransient__(df_feat_nont, task, min_obs, num_features)
    return df_feat_nont

def load_transient_catalog():
    filename = 'transient_catalog.pickle'
    indir = DATA_PATH; filepath = indir + filename
    df_cat = pd.read_pickle(filepath)
    # Rename columns to match light curves
    df_cat = df_cat.rename(columns={'TransientID': 'ID', 'Classification': 'class'})
    df_cat.ID = 'TranID' + df_cat.ID.apply(str)
    df_cat = df_cat.set_index('ID')
    return df_cat

def split(df, remove_ids=True, oversampled=False):
    np.random.seed(42)
#     # Remove IDs
#     if remove_ids: df = df.drop(['ID'], axis=1)
    # Obtain X and y
    X = df.drop(['class'], axis=1).as_matrix()
    print(X[:1])
    y = df['class'].as_matrix()
    if oversampled: return X, None, y, None
    # Count total number of objects
#    print('Total number of objects: ', np.sum(np.unique(y, return_counts=True)[1]))
    # Count number of objects per class
#    print('Number of objects per class:\n', dict(zip(*np.unique(y, return_counts=True))), '\n')
    # Split in Test & Train Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def binary(min_obs, num_features, remove_ids=True, oversample=False):
    # Load feature dataframes
    df_feat_tran = load_transient_features(min_obs, num_features, oversample)
    df_feat_nont = load_nontransient_features(min_obs, num_features, oversample)
    
    # Add output class '1' to transient objects
    df_feat_tran['class'] = 'transient'
    # Add output class '0' to non-transient objects
    df_feat_nont['class'] = 'non-transient'
    # Merge dataframes
    df = df_feat_tran.append(df_feat_nont, ignore_index=True)
    
    X_train, X_test, y_train, y_test = split(df, remove_ids, oversample)
    if oversample: _, X_test, _, y_test = binary(min_obs, num_features, remove_ids)
    return X_train, X_test, y_train, y_test

def six_transient(min_obs, num_features, remove_ids=True, oversample=False):
    np.random.seed(42)
    df_cat = load_transient_catalog()
    df_feat_tran = load_transient_features(min_obs, num_features, oversample)

    # Add classs label to transient objects
    df_feat_tran = df_feat_tran.join(df_cat, how='inner')
    # Remove ambiguous classes
    df = df_feat_tran[df_feat_tran['class'].isin(top_transient_classes())]
    
    X_train, X_test, y_train, y_test = split(df, remove_ids, oversample)
    if oversample: _, X_test, _, y_test = six_transient(min_obs, num_features, remove_ids)
    return X_train, X_test, y_train, y_test

def seven_transient(min_obs, num_features, remove_ids=True, oversample=False):
    np.random.seed(42)
    df_cat = load_transient_catalog()
    df_feat_tran = load_transient_features(min_obs, num_features, oversample)
    # Add classs label to transient objects
    df_feat_tran = df_feat_tran.join(df_cat, how='inner')
    # Remove ambiguous classes
    in_top = lambda row: ('Other' if row['class'] not in top_transient_classes() else row['class'])
    df['class'] = df.apply( in_top , axis=1)    
    
    X_train, X_test, y_train, y_test = split(df, remove_ids, oversample)
    if oversample: _, X_test, _, y_test = seven_transient(min_obs, num_features, remove_ids)
    return X_train, X_test, y_train, y_test
    
def seven_class(min_obs, num_features, remove_ids=True, oversample=False):
    np.random.seed(42)
    df_cat = load_transient_catalog()
    df_feat_tran = load_transient_features(min_obs, num_features, oversample)
    df_feat_nont = load_nontransient_features(min_obs, num_features, oversample)
    # Add classs label to transient objects
    df_feat_tran = df_feat_tran.join(df_cat, how='inner')
    # Remove ambiguous classes
    df_feat_tran = df_feat_tran[df_feat_tran['class'].isin(top_transient_classes())]
    # Add class to non-transient features
    df_feat_nont['class'] = 'Non-Transient'
    # Sample non-transients features as big as largest class
    big_class_size = df_feat_tran.groupby('class')['ID'].count().max()
    IDs = np.random.choice(df_feat_nont.ID.unique(), size=big_class_size, replace=False)
    df_feat_nont = df_feat_nont[df_feat_nont.ID.isin(IDs)]
    # Merge transient and non-transient df
    df = df_feat_tran.append(df_feat_nont, ignore_index=True)

    X_train, X_test, y_train, y_test = split(df, remove_ids, oversample)
    if oversample: _, X_test, _, y_test = seven_class(min_obs, num_features, remove_ids)
    return X_train, X_test, y_train, y_test

def eight_class(min_obs, num_features, remove_ids=True, oversample=False):
    np.random.seed(42)

    df_cat = load_transient_catalog()
    df_feat_tran = load_transient_features(min_obs, num_features, oversample)
    df_feat_nont = load_nontransient_features(min_obs, num_features, oversample)

    # Add classs label to transient objects
    df_feat_tran = df_feat_tran.join(df_cat, how='inner')
    # Remove ambiguous classes
    in_top = lambda row: ('Other' if row['class'] not in top_transient_classes() else row['class'])
    df_feat_tran['class'] = df_feat_tran.apply( in_top , axis=1)
    # Add class to non-transient features
    df_feat_nont['class'] = 'Non-Transient'
    # Sample non-transients features as big as largest class
    big_class_size = df_feat_tran.groupby('class')['ID'].count().max()
    IDs = np.random.choice(df_feat_nont.ID.unique(), size=big_class_size, replace=False)
    df_feat_nont = df_feat_nont[df_feat_nont.ID.isin(IDs)]
    # Merge transient and non-transient df
    df = df_feat_tran.append(df_feat_nont, ignore_index=True)
    
    X_train, X_test, y_train, y_test = split(df, remove_ids, oversample)
    if oversample: _, X_test, _, y_test = eight_class(min_obs, num_features, remove_ids)
    return X_train, X_test, y_train, y_test


def load_oversampled_transient_features(min_obs, num_features):
    # Loead transient features
    indir = FEATURES_PATH
    filename = 'overs_transient_{}obs_{}feats.pickle'.format(min_obs, num_features)  
    inpath = indir + filename
    df_feat_tran = pd.read_pickle(inpath)
    return df_feat_tran

def load_oversampled_nontransient_features(min_obs, num_features):
    # Load nontransient Features
    indir = FEATURES_PATH
    filename = 'overs_nontransient_{}obs_{}feats.pickle'.format(min_obs, num_features)  
    inpath = indir + filename
    df_feat_nont = pd.read_pickle(inpath)
    return df_feat_nont                         
    
def __filter_oversampled_transient__(df_all, task, min_obs, num_features):
    np.random.seed(42)
#    print('Loading transient')
    df = df_all
#    print(df.CopyID.head())
    # Obtain X data for given task
    X_train, X_test, y_train, y_test = task(min_obs, num_features, remove_ids=False)
    # Obtain tranining IDs for given task
    train_ids = X_train[:,0]
    # Remove copy number from ids
#     df['Copy'] = df.CopyID.str[0]
#     df['ID'] = df.CopyID.str[2:]
    df['class'] = ''
    # Remove testing data from df
    print(X_train[:2])
    df_task = df.loc[~df.index.isin(train_ids)].copy()
#    print('df_task', df_task.shape)
    # Count number of objects per class
    class_count = dict(zip(*np.unique(y_train, return_counts=True)))
    max_count = max(class_count.values())
    # Define oversampling indexes
    index_list = np.array([])
#    print(class_count)
    for k, c in class_count.items():
#        print(k, is_transient)
        if k.lower() == 'non-transient': continue
        class_ids = X_train[np.where(y_train[:]==k),0][0]
        class_df =  df_task[df_task.ID.isin(class_ids)]
        last_object = class_df.groupby(['ID', 'Copy'], sort=False, as_index=False).mean().iloc[max_count]
        last_index = class_df[(class_df.ID == last_object.ID) & (class_df.Copy == last_object.Copy)].index[0]
        class_indexes = class_df.loc[:last_index-1].index
        df_task.loc[class_indexes, 'class'] = k
        index_list = np.concatenate((index_list, class_indexes), axis=0)
    # Use balanced oversampled data
    df_task = df_task.loc[index_list]
#    print(df_task.CopyID.unique().shape)
#    df_task['ID'] = pd.to_numeric(df_task.CopyID.str[8:])
    df_task = df_task.drop(columns=['CopyID', 'Copy', 'class'])
    return df_task

def __filter_oversampled_nontransient__(df_all, task, min_obs, num_features):
    #print('Loading non-transient')
    df = df_all.copy()
#    print(df.CopyID.head())
    # Obtain X data for given task
    X_train, X_test, y_train, y_test = task(min_obs, num_features, remove_ids=False)
    # Obtain tranining IDs for given task
    train_ids = X_train[:,0]
    # Remove copy number from ids
    df['Copy'] = df.CopyID.str[0]
    df['ID'] = df.CopyID.str[2:]
    # Remove testing data from df
    df_task = df[~df.ID.isin(train_ids)]
#    print('df_task', df_task.shape)
    # Count number of objects per class
    class_count = dict(zip(*np.unique(y_train, return_counts=True)))
    max_count = max(class_count.values())
    df_task = df_task.iloc[:max_count]
#    print(df_task.CopyID.unique().shape)
#    df_task['ID'] = pd.to_numeric(df_task.CopyID.str[8:])
    df_task = df_task.drop(columns=['CopyID', 'Copy'])
    return df_task
    