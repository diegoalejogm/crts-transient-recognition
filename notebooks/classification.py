import os
import inputs2
import helpers
import itertools
from sklearn.externals import joblib
import pandas as pd
import numpy as np
# import pandas as pd

from sklearn.svm import SVC
# from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # , cross_val_score

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix  # , classification_report


RESULTS_PATH = '../results/'


def svc():
    params = {
        'kernel': ['rbf'],
        'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        'C': [1, 10, 100, 1000]
    }
    return SVC(random_state=0, class_weight='balanced'), params


def rf():
    params = {
        'n_estimators': [200, 700],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    return RandomForestClassifier(random_state=0, class_weight='balanced'), params


def mlp():
    params = {
        'learning_rate': ['constant', "adaptive"],
        'hidden_layer_sizes': [(100), (100, 100)],
        'alpha': [1e-1, 1e-2, 1e-3, 1e-4],
        'activation': ["logistic", "relu"]
    }
    return MLPClassifier(random_state=0, max_iter=2000), params


def _task_name_(input_func):
    return input_func.__name__.replace('load_', '')


def scorers():
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='weighted'),
               'recall': make_scorer(recall_score, average='weighted'),
               'f1_score': make_scorer(f1_score, average='weighted')
               }
    return scoring


def _confusion_matrix_(y_true, y_pred, normalized=True):
    all_labels = np.unique(y_true)
    cnf_matrix = confusion_matrix(y_true, y_pred, all_labels)
    if normalized:
        cnf_matrix = cnf_matrix.astype(
            'float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        cnf_matrix = np.around(cnf_matrix, decimals=4) * 100
#     print(classification_report(y_true, y_pred, all_labels, digits=6))
    return cnf_matrix, all_labels


def _format_test_scores_(scores_list, weighted):
    assert weighted is not None

    precision, recall, fscore, support = scores_list
    if weighted:
        return {'test_precision': precision, 'test_recall': recall,
                'test_fscore': fscore}
    else:
        return {'test_precision_by_class': precision, 'test_recall_by_class': recall,
                'test_fscore_by_class': fscore, 'test_support_by_class': support}


def _combination_results_dict_(combination, clf, test_scores_dict, test_scores_by_class_dict, cnf_matrix, class_labels):
    min_obs, num_features, oversample, model, scaler = combination
    model_name = model.__name__
    scaler_name = scaler.__name__
    params_and_train_scores = {
        'min_obs': min_obs, 'num_features': num_features, 'oversample': oversample,
        'model': model_name, 'scaler': scaler_name,
        'train_best_score': clf.best_score_, 'best_params': clf.best_params_,
        'cnf_matrix': cnf_matrix, 'class_labels': class_labels
    }
    combination_results = {
        **params_and_train_scores,
        **test_scores_dict,
        **test_scores_by_class_dict
    }
    return combination_results


def _combination_path_(combination, type_name):
    min_obs, num_features, oversample, model, scaler = combination
    is_oversampled = '' if oversample else 'non-'
    path = RESULTS_PATH + type_name + '/'
    path += '{}obs/{}feats/{}oversample/{}/{}'.format(
        min_obs, num_features, is_oversampled, model.__name__, scaler.__name__)
    return path


def _test_combination_results_(combination, clf, inputs, task_name):
    # Predict Test samples with Classifier
    _, _, X_test, y_test = inputs
    y_pred = clf.predict(X_test)
    # Generate Confusion Matrix
    cnf_matrix, class_labels = _confusion_matrix_(y_test, y_pred)
    # Combination Path
    # dirr = _combination_path_(
    # combination, 'confusion_matrices')
    # Store Confusion Matrix
    # helpers.make_dir_if_not_exists(dirr)
    # np.save(dirr + '/{}'.format(task_name), cnf_matrix)
    # np.save(dirr + '/{}_labels'.format(task_name), labels)
    # Generate Scores Dicts (don't save them)

    scores = precision_recall_fscore_support(
        y_test, y_pred, average='weighted')
    scores_by_class = precision_recall_fscore_support(y_test, y_pred)
    scores_dict = _format_test_scores_(scores, weighted=True)
    scores_by_class_dict = _format_test_scores_(
        scores_by_class, weighted=False)
    # Return dict
    return _combination_results_dict_(
        combination, clf, scores_dict, scores_by_class_dict, cnf_matrix, class_labels
    )


def _store_task_results_(task_results, task_models, task_name):
    # Create DataFrame
    df_results = pd.DataFrame(task_results)
    # Store DataFrame with current task's results
    dirr = RESULTS_PATH + 'dataframes/'
    helpers.make_dir_if_not_exists(dirr)
    df_results.to_pickle(dirr + '{}.pkl'.format(task_name))

    # Find best model and store it
    best_model_index = df_results['test_fscore'].idxmax()
    best_model = task_models[best_model_index]
    dirr = RESULTS_PATH + 'models/'
    # print(best_model_index)
    helpers.make_dir_if_not_exists(dirr)
    joblib.dump(best_model, dirr + task_name)

    return df_results, best_model


def _should_run_combination_(task_input, combination):
    _, _, oversample, _, _ = combination
    return task_input != inputs2.load_binary or not oversample


def _train_model_(inputs, model_func):
    # Split inputs
    X_train, y_train, _, _ = inputs
    # Obtain Model
    model, params = model_func()
    # Train & cross-validate
    grid_search = GridSearchCV(model, params, cv=StratifiedKFold(2), scoring=scorers(),
                               refit='f1_score', return_train_score=True)
    grid_search.fit(X_train, y_train)
    # Copy classifier
    clf = grid_search
    return clf


def _run_single_task_(task_input, param_combinations):
    task_name = _task_name_(task_input)
    task_results = list()
    task_models = list()
    # For each parameter combination
    for combination in param_combinations:
        should_run_task = _should_run_combination_(task_input, combination)
        if not should_run_task:
            continue

        # Pre: Load variables
        min_obs, num_features, oversample, model, scaler = combination
        print('Starting: {}, {}, {}, {}, {}, {}'.format(
            task_name, min_obs, num_features, oversample, model.__name__,
            scaler.__name__))

        # Obtain inputs
        inputs = task_input(min_obs, num_features, oversample, scaler)
        # Train-Validate model
        clf = _train_model_(inputs, model)
        # Store Classifier Results on Test Data
        test_combination_results_dict = _test_combination_results_(
            combination, clf, inputs, task_name)

        # Append task results
        task_results.append(test_combination_results_dict)
        task_models.append(clf.best_estimator_)
        print('Finished combination')

    # Store Task Results
    task_results_df, _ = _store_task_results_(
        task_results, task_models, task_name)

    return task_results_df


def run_tasks(inputs_list, num_obs_list, num_features_list, oversample_list, model_list, scaler_list):
    all_task_results = []
    # For each task
    for task_input in inputs_list:
        param_combinations = itertools.product(
            num_obs_list, num_features_list, oversample_list, model_list, scaler_list
        )
        task_results = _run_single_task_(task_input, param_combinations)
        if task_results is not None:
            all_task_results.append(task_results)
    return all_task_results
