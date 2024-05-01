from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor
from functools import partial
import time
from multiprocessing import Pool


def tune_rf_parameters(final_training_sets, k):
    param_grid = {
        'n_estimators': [20, 30, 50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10, 15],
    }
    model = RandomForestRegressor()
    best_params = []

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=None)

    # for i in range(k):
    X_train = final_training_sets[k][0]
    y_train = final_training_sets[k][1].values.ravel()
    grid_search.fit(X_train, y_train)

    best_params.append(grid_search.best_params_)

    print("Best Parameters for rf:")
    print(grid_search.best_params_)

    return best_params


def tune_bagging_parameters(final_training_sets, k):
    param_grid = {
        'n_estimators': [30, 50, 100, 200],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False]
    }
    model = BaggingRegressor()
    best_params = []

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=None)

    # for i in range(k):
    X_train = final_training_sets[k][0]
    y_train = final_training_sets[k][1].values.ravel()
    grid_search.fit(X_train, y_train)

    best_params.append(grid_search.best_params_)

    print("Best Parameters for bagging:")
    print(grid_search.best_params_)

    return best_params


def tune_boosting_parameters(final_training_sets, k):
    param_grid = {
        'n_estimators': [5, 10, 20, 30, 50, 100],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [3, 4, 5],
    }
    model = GradientBoostingRegressor()
    best_params = []

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid)

    X_train = final_training_sets[k][0]
    y_train = final_training_sets[k][1].values.ravel()
    grid_search.fit(X_train, y_train)

    best_params.append(grid_search.best_params_)

    print("Best Parameters for boosting:")
    print(grid_search.best_params_)
    X_train = final_training_sets[k][0]
    y_train = final_training_sets[k][1].values.ravel()
    grid_search.fit(X_train, y_train)

    best_params.append(grid_search.best_params_)

    print("Best Parameters:")
    print(grid_search.best_params_)

    return best_params


def tune_cnn_parameters(standarized_filled_training_sets, k):
    param_grid = {
        'dense_units': [16, 32, 64],
        'activation': ['relu', 'sigmoid'],
        'optimizer': ['adam', 'sgd'],
        'loss': ['mse', 'mae']
        }
    best_params = []

    X_train = standarized_filled_training_sets[k][0]
    y_train = standarized_filled_training_sets[k][1].values.ravel()

    model = Sequential()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

    grid_search.fit(X_train, y_train)

    best_params.append(grid_search.best_params_)

    print("Best Parameters:")
    print(grid_search.best_params_)

    return best_params

def tune_xgboost_parameters(final_training_sets, k):
    param_grid = {
        'n_estimators': [10, 20, 100, 200, 300, 400],
        'max_depth': [3, 5, 7, None],
        'learning_rate': [0.1, 0.01, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }
    model = XGBRegressor()
    best_params = []

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=None)

    X_train = final_training_sets[k][0]
    y_train = final_training_sets[k][1].values.ravel()
    grid_search.fit(X_train, y_train)

    best_params.append(grid_search.best_params_)

    print("Best Parameters for xgboost:")
    print(grid_search.best_params_)

    return best_params

def helper_function(inversed_filled_training_sets, k):
    start_time = time.time()
    print(f"Starting process with k={k} at {time.strftime('%H:%M:%S')}")
    result = tune_xgboost_parameters(inversed_filled_training_sets, k)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process with k={k} completed in {elapsed_time} seconds.")
    return result