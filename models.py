from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def model_boost(final_training_sets, final_testing_sets, k, n, max_depth, max_features, min_samples_leaf, min_samples_split, n_estimators):
    model = GradientBoostingRegressor(max_depth = max_depth, max_features = max_features, min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, n_estimators = n_estimators)

    for i in range(k, n):
        X_train = final_training_sets[i][0]
        y_train = final_training_sets[i][1].values.ravel()
        model.fit(X_train, y_train)

    for i in range(k, n):
        X_test = final_testing_sets[i][0]
        y_test = final_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)

    feature_importances = []
    for i in range(k, n):
        X_test = final_testing_sets[i][0]
        feature_importances.append(model.feature_importances_)
    median_importances = np.median(feature_importances, axis=0)
    variable_names = final_training_sets[k][0].columns

    MAPE = []

    for i in range(k, n):
        X_test = final_testing_sets[i][0]
        y_test = final_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)
        error = np.mean(np.abs((y_test - predictions) / y_test))
    
        MAPE.append(error)

    median_error = np.median(MAPE)
    print("\nboosting")
    print(f"Median Error: {median_error}")

    # print("Median Importance:")
    # sorted_importances = sorted(zip(variable_names, median_importances), key=lambda x: x[1], reverse=True)
    # for name, importance in sorted_importances:
    #     print(f"{name}: {importance}")

    return model, MAPE, median_importances, variable_names


def model_rf(final_training_sets, ifinal_testing_sets, k, n, n_estimators, max_depth, min_samples_split):
    model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split)

    for i in range(k, n):
        X_train = final_training_sets[i][0]
        y_train = final_training_sets[i][1].values.ravel()
        model.fit(X_train, y_train)

    for i in range(k, n):
        X_test = ifinal_testing_sets[i][0]
        y_test = ifinal_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)

    MAPE = []
    for i in range(k, n):
        X_test = ifinal_testing_sets[i][0]
        y_test = ifinal_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)
        error = np.mean(np.abs((y_test - predictions) / y_test))
        MAPE.append(error)

    median_error = np.median(MAPE)
    print("\nrandom forest")
    print(f"Median Error: {median_error}")
    return model, MAPE


def model_bagging(final_training_sets, final_testing_sets, k, n, bootstrap, bootstrap_features, max_features, max_samples, n_estimators):
    model = BaggingRegressor(bootstrap = bootstrap, bootstrap_features= bootstrap_features, max_features = max_features, max_samples = max_samples, n_estimators = n_estimators)

    for i in range(k, n):
        X_train = final_training_sets[i][0]
        y_train = final_training_sets[i][1].values.ravel()
        model.fit(X_train, y_train)

    for i in range(k, n):
        X_test = final_testing_sets[i][0]
        y_test = final_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)
    
    MAPE = []
    for i in range(k, n):
        X_test = final_testing_sets[i][0]
        y_test = final_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)
        error = np.mean(np.abs((y_test - predictions) / y_test))
        MAPE.append(error)
        #print(f"Error for model {i+1}: {error}%")

    median_error = np.median(MAPE)
    print("\nbagging")
    print(f"Median Error: {median_error}")

    return model, MAPE


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

    MAPE = []
    for i in range(k, n):
        X_test = standarized_filled_testing_sets[i][0]
        y_test = standarized_filled_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)
        error = np.mean(np.abs((y_test - predictions) / y_test))
        MAPE.append(error)

    median_error = np.median(MAPE)
    print("\n cnn")
    print(f"Median Error: {median_error}")

    return model, MAPE


def model_xgboost(final_training_sets, final_testing_sets, k, n, max_depth, learning_rate, subsample, colsample_bytree, n_estimators):
    model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, n_estimators=n_estimators)

    for i in range(k, n):
        X_train = final_training_sets[i][0]
        y_train = final_training_sets[i][1].values.ravel()
        model.fit(X_train, y_train)

    feature_importances = []
    for i in range(k, n):
        X_test = final_testing_sets[i][0]
        feature_importances.append(model.feature_importances_)
    median_importances = np.median(feature_importances, axis=0)
    variable_names = final_training_sets[k][0].columns

    MAPE = []

    for i in range(k, n):
        X_test = final_testing_sets[i][0]
        y_test = final_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)
        error = np.mean(np.abs((y_test - predictions) / y_test))
        MAPE.append(error)

    r2_scores = []
    for i in range(k, n):
        X_test = final_testing_sets[i][0]
        y_test = final_testing_sets[i][1].values.ravel()
        predictions = model.predict(X_test)
        r2_score = model.score(X_test, y_test)
        r2_scores.append(r2_score)

    median_r2_score = np.median(r2_scores)
    # print("\nR2 Scores")
    # print(f"Median R2 Score: {median_r2_score}")
    median_error = np.median(MAPE)
    print("\nXGBoost")
    print(f"Median Error: {median_error}")

    return model, MAPE, median_importances, variable_names, r2_scores