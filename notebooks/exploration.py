import features
import numpy as np
import pandas as pd
import classification as c
import matplotlib
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier


DATAFRAMES_PATH = c.RESULTS_PATH + 'dataframes/'


def _filename_(task_index):
    filenames = ['binary', 'six_transients', 'seven_transients',
                 'seven_classes', 'eight_classes']
    return filenames[task_index]


def _load_dataframe_(task_index):
    file_path = DATAFRAMES_PATH + _filename_(task_index) + '.pkl'
    return pd.read_pickle(file_path)


def _model_(task_index):
    filename = _filename_(task_index)
    clf = joblib.load('../results/models/{}'.format(filename))
    return clf


def _top_(index):
    df = _load_dataframe_(index)
    return df.sort_values('test_fscore', ascending=False).iloc[0]


def print_confusion_matrix(top_series):
    print(top_series.class_labels)
    print(top_series.cnf_matrix)
    print('Support', top_series.test_support_by_class)


def print_classification_results(top_series):
    print('Score:', top_series.test_fscore)
    print(top_series.class_labels)
    print(top_series.test_support_by_class)
    print('FScores:', top_series.test_fscore_by_class)
    print('Precisions:', top_series.test_precision_by_class)
    print('Recalls:', top_series.test_recall_by_class)


##

def feature_importance(clf):
    assert type(clf) is RandomForestClassifier
    importances = clf.feature_importances_
    num_features = importances.shape[0]
    indices = np.argsort(importances)[::-1]

    # Store in list as tuples
    sorted_features = []
    for f in range(num_features):
        feature_index = indices[f]
        feature_name = features.ALL_NO_CLASS[feature_index]
        current_feature_importance = importances[feature_index]
        sorted_features.append((feature_name, current_feature_importance))

    return sorted_features


def print_feature_importance(importance_list):
    print("Feature ranking:")
    for i, f in enumerate(importance_list):
        print("{}. feature {} ({:.2%})".format(i + 1, f[0], f[1]))


# LOAD TOP SERIES AND MODEL

def _top_model_(index):
    return _top_(index), _model_(index)


def top_binary():
    return _top_model_(0)


def top_six_t():
    return _top_model_(1)


def top_seven_t():
    return _top_model_(2)


def top_seven_c():
    return _top_model_(3)


def top_eight_c():
    return _top_model_(4)
