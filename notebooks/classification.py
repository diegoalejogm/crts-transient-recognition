RESULTS_PATH = '../results/'

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_validate_test_model(X_train, y_train, X_test, y_test, model, params, min_obs, num_features, oversample, task, scaler):
    # Precision decimal places
    digits = 4
    # Train & cross-validate
    grid_search = GridSearchCV(model, params, cv=StratifiedKFold(2))
    grid_search.fit(X_train, y_train)
    # Train new model with all train data
    clf = grid_search.best_estimator_
    clf.fit(X_train, y_train)
    # Predict test inputs with new model
    y_pred = clf.predict(X_test)
    # Create results using real and predicted labels of test data 
    results_str = results_string(X_train, y_train, X_test, y_test, y_pred, grid_search, digits=digits)
    print(results_str)
    # Save results
    task_name = task.__name__
    model_name = model.__class__.__name__
    scaler_name = scaler.__class__.__name__
    filename = '{}_{}obs_{}feat_{}_{}'.format(task_name, min_obs, num_features, model_name, scaler_name)
    if oversample: filename = 'oversample_' + filename
    with open(RESULTS_PATH + filename + '.txt', 'w+') as f: f.write(results_str)
        
    return clf

def results_string(X_train, y_train, X_test, y_test, y_pred, grid_search, digits):
    float_param = '{0:.' + str(digits) + 'f}'
    results = str()
    results += 'Train Shapes (X, y):  {}, {}\n'.format(X_train.shape, y_train.shape)
    results += 'Test Shapes (X, y):  {}, {}\n'.format(X_test.shape, y_test.shape)
    uniques = np.unique(y_train, return_counts=True)
    results += 'Unique classes: {}\n'.format(uniques[0])
    results += 'Unique count: {}\n'.format(uniques[1])
    results += 'Best Params: {}\n'.format(grid_search.best_params_)
    results += ('Validation Accuracy: ' + float_param + '\n').format(grid_search.best_score_)
    results += ('Test Accuracy: ' + float_param + '\n').format(accuracy_score(y_test, y_pred))
    results += 'Report:\n {}'.format(classification_report(y_test, y_pred, digits=digits))
    results += 'Confusion Matrix:\n {}\n'.format(clf_confusion_matrix(y_pred, y_test))
    results += 'Normalized Confusion Matrix:\n {}'.format(clf_confusion_matrix(y_pred, y_test, True))
    return results

def clf_confusion_matrix(y_pred, y_true, normalized=False):
    all_labels = np.unique(y_true)
    cnf_matrix = confusion_matrix(y_true, y_pred, all_labels)
    if normalized:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        cnf_matrix = np.around(cnf_matrix, decimals=4) * 100
    return cnf_matrix

def svc(X_train, y_train, X_test, y_test, min_obs, num_features, oversample, task, scaler):
    params = {
    'kernel': ['rbf'],
    'gamma':[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    'C': [1, 10, 100, 1000]
    }
    model = SVC(random_state=0, class_weight='balanced')
    clf1 = train_validate_test_model(X_train, y_train, X_test, y_test, model, params, min_obs, num_features, oversample, task, scaler)
    
def rf(X_train, y_train, X_test, y_test, min_obs, num_features, oversample, task, scaler):
    params = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
    }
    model = RandomForestClassifier(random_state=0, class_weight='balanced')
    clf2 = train_validate_test_model(X_train, y_train, X_test, y_test, model, params, min_obs, num_features, oversample, task, scaler)
    
def mlp(X_train, y_train, X_test, y_test, min_obs, num_features, oversample, task, scaler):
    params = {
    'learning_rate': ['constant', "adaptive"],
    'hidden_layer_sizes': [(100), (100,100)],
    'alpha': [1e-1, 1e-2, 1e-3, 1e-4],
    'activation': ["logistic", "relu"]
    }
    model = MLPClassifier(random_state=0)
    clf3 = train_validate_test_model(X_train, y_train, X_test, y_test, model, params, min_obs, num_features, oversample, task, scaler)