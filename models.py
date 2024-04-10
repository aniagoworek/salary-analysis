from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def tune_bagging_hyperparameters(inversed_filled_training_sets, k, n):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False]
    }
    model = BaggingRegressor()
    best_params = []

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=n-k)

    for i in range(k, n):
        X_train = inversed_filled_training_sets[i][0]
        y_train = inversed_filled_training_sets[i][1].values.ravel()
        grid_search.fit(X_train, y_train)

        best_params.append(grid_search.best_params_)

        print("Best Parameters:")
        print(grid_search.best_params_)

    return best_params


def tune_rf_hyperparameters(inversed_filled_training_sets, k, n):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15, 20],
    }
    model = RandomForestRegressor()
    best_params = []

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=n-k)

    for i in range(k, n):
        X_train = inversed_filled_training_sets[i][0]
        y_train = inversed_filled_training_sets[i][1].values.ravel()
        grid_search.fit(X_train, y_train)

        best_params.append(grid_search.best_params_)

        print("Best Parameters:")
        print(grid_search.best_params_)

    return best_params


def find_optimal_hyperparameters(inversed_filled_training_sets, k, n):
    param_grid = {
        'n_estimators': range(30, 301),
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    models = GradientBoostingRegressor()
    best_params = []

    for model in models:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=n-k)

        for i in range(k, n):
            X_train = inversed_filled_training_sets[i][0]
            y_train = inversed_filled_training_sets[i][1].values.ravel()
            grid_search.fit(X_train, y_train)

        best_params.append(grid_search.best_params_)

        print("Best Parameters:")
        print(grid_search.best_params_)

    return best_params


def model_rf(inversed_filled_training_sets, inversed_filled_testing_sets, k, n, n_estimators, max_depth, min_samples_split):
    model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split)

    for i in range(k, n):
        X_train = inversed_filled_training_sets[i][0]
        y_train = inversed_filled_training_sets[i][1].values.ravel()
        model.fit(X_train, y_train)

    for i in range(k, n):
        X_test = inversed_filled_testing_sets[i][0]
        y_test = inversed_filled_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)

    errors = []
    for i in range(k, n):
        X_test = inversed_filled_testing_sets[i][0]
        y_test = inversed_filled_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)
        error = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        errors.append(error)

    median_error = np.median(errors)
    print("\nrandom forest")
    print(f"Median Error: {median_error}%")
    return model


def model_boost(inversed_filled_training_sets, inversed_filled_testing_sets, k, n):
    model = GradientBoostingRegressor()

    for i in range(k, n):
        X_train = inversed_filled_training_sets[i][0]
        y_train = inversed_filled_training_sets[i][1].values.ravel()
        model.fit(X_train, y_train)

    for i in range(k, n):
        X_test = inversed_filled_testing_sets[i][0]
        y_test = inversed_filled_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)

    feature_importances = []
    for i in range(k, n):
        X_test = inversed_filled_testing_sets[i][0]
        feature_importances.append(model.feature_importances_)
    median_importance = np.median(feature_importances, axis=0)
    variable_names = inversed_filled_training_sets[k][0].columns
    print("\nFeature Importances - boosting:")
    sorted_importances = sorted(zip(variable_names, median_importance), key=lambda x: x[1], reverse=True)
    for name, importance in sorted_importances:
        print(f"{name}: {importance}")

        # Feature Importances - boosting
        plt.figure(figsize=(10, 6))
        plt.bar(variable_names, median_importance)
        plt.title('Feature Importances - Boosting')
        plt.xlabel('Variable')
        plt.ylabel('Importance')
        plt.xticks(rotation=90)
        plt.savefig(f'feature_importance_plot{i - k + 2}.svg', format='svg')
        plt.show()

    errors = []
    for i in range(k, n):
        X_test = inversed_filled_testing_sets[i][0]
        y_test = inversed_filled_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)
        error = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        errors.append(error)

    # Boxplot for errors
    import matplotlib.pyplot as plt
    plt.boxplot(errors)
    plt.title('Error Boxplot')
    plt.ylabel('Error (%)')
    plt.show()

    median_error = np.median(errors)
    print("\nboosting")
    print(f"Median Error: {median_error}%")
    return model


def model_bagging(inversed_filled_training_sets, inversed_filled_testing_sets, k, n, bootstrap, bootstrap_features, max_features, max_samples, n_estimators):
    model = BaggingRegressor(bootstrap = bootstrap, bootstrap_features= bootstrap_features, max_features = max_features, max_samples = max_samples, n_estimators = n_estimators)

    for i in range(k, n):
        X_train = inversed_filled_training_sets[i][0]
        y_train = inversed_filled_training_sets[i][1].values.ravel()
        model.fit(X_train, y_train)

    for i in range(k, n):
        X_test = inversed_filled_testing_sets[i][0]
        y_test = inversed_filled_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)
    
    errors = []
    for i in range(k, n):
        X_test = inversed_filled_testing_sets[i][0]
        y_test = inversed_filled_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)
        error = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        errors.append(error)
        #print(f"Error for model {i+1}: {error}%")

    median_error = np.median(errors)
    print("\nbagging")
    print(f"Median Error: {median_error}%")

    return model


def model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k , n):
    input_shape = standarized_filled_training_sets[k][0].shape[1]
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    for i in range(k, n):
        X_train = standarized_filled_training_sets[i][0]
        y_train = standarized_filled_training_sets[i][1].values.ravel()
        model.fit(X_train, y_train)

    for i in range(k, n):
        X_test = standarized_filled_testing_sets[i][0]
        y_test = standarized_filled_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)


    errors = []
    for i in range(k, n):
        X_test = standarized_filled_testing_sets[i][0]
        y_test = standarized_filled_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)
        error = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        errors.append(error)

    median_error = np.median(errors)
    print("\n cnn")
    print(f"Median Error: {median_error}%")

    return model


