import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.calibration import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer


def remove_sets_with_many_missing_values2(testing_sets):
    cleaned_testing_sets = []

    for X_test, y_test in testing_sets:
        missing_values_percentage = (X_test.isnull().sum().sum() + y_test.isnull().sum().sum()) / (X_test.size + y_test.size)
        if missing_values_percentage <= 0.20:
            cleaned_testing_sets.append((X_test, y_test))

    removed_sets = list(set(range(len(testing_sets))) - set(range(len(cleaned_testing_sets))))
    # print(f'Removed sets: {removed_sets}')
    return cleaned_testing_sets

def standardize_data2(testing_sets, scalers):
    standardized_testing_sets = []

    for i in range(len(testing_sets)):
        X_test, y_test = testing_sets[i]
        X_scaler, y_scaler = scalers[i]
        X_test = X_scaler.transform(testing_sets[i][0])
        y_test = y_scaler.transform(testing_sets[i][1].values.reshape(-1, 1)).flatten()  # Use the original y_train as the target
        standardized_testing_sets.append((X_test, y_test))

    # print(f'\nStandaryzowane dane X: {standardized_testing_sets[6][0]}') # pierwsza liczba: 0-29 = lata 2016-2017, 30-59 = lata 2016-2018, 60-89 = lata 2016-2019, 90-119 = lata 2016-2020, 120-149 = lata 2016-2021, 150-179 = lata 2016-2022
    # print(f'\nStandaryzowane dane y: {standardized_testing_sets[6][1]}') # pierwsza liczba: 0-29 = lata 2016-2017, 30-59 = lata 2016-2018, 60-89 = lata 2016-2019, 90-119 = lata 2016-2020, 120-149 = lata 2016-2021, 150-179 = lata 2016-2022
    return standardized_testing_sets

# Fill missing values in standardized_training_sets using KNNImputer
def fill_blanks_with_knn2(standardized_testing_sets, optimal_k):
    imputer = KNNImputer(n_neighbors=optimal_k)
    filled_testing_sets = []

    for i in range(len(standardized_testing_sets)):
        X_test, y_test = standardized_testing_sets[i]
        X_test_filled = imputer.fit_transform(X_test)
        y_test_filled = imputer.fit_transform(y_test.reshape(-1, 1)).flatten()
        filled_testing_sets.append((X_test_filled, y_test_filled))
    # print(f'\nWypelnione przez knn X: {filled_testing_sets[6][0]}')
    # print(f'\nWypelnione przen knn y: {filled_testing_sets[6][1]}')
    return filled_testing_sets

# Perform cross-validation to evaluate the performance of KNN imputation
def cross_validate_filled_sets2(filled_testing_sets):
    mse_values = []

    for _, (X_test, y_test) in enumerate(filled_testing_sets):
        knn = KNeighborsRegressor(n_neighbors=1)
        y_pred = cross_val_predict(knn, X_test, y_test, cv=50)
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)

    # Plot the MSE values
    # plt.plot(range(1, len(filled_testing_sets) + 1), mse_values)
    # plt.xlabel('Set')
    # plt.ylabel('Mean Squared Error')
    # plt.title('MSE for each set')
    # plt.show()
    return mse_values

# Inverse standardization of filled_training_sets
def inverse_standardization2(filled_testing_sets, testing_sets, scalers):
    inverse_filled_testing_sets = []

    for i in range(len(filled_testing_sets)):
        X_test, y_test = filled_testing_sets[i]
        X_test_inverse = scalers[i][0].inverse_transform(X_test)
        y_test_inverse = scalers[i][1].inverse_transform(y_test.reshape(-1, 1)).flatten()
        inverse_filled_testing_sets.append((pd.DataFrame(X_test_inverse, index=testing_sets[i][0].index), pd.DataFrame(y_test_inverse, index=testing_sets[i][1].index)))

    inversed_testing_sets = []

    for i in range(len(inverse_filled_testing_sets)):
        X_test_inverse, y_test_inverse = inverse_filled_testing_sets[i]
        X_test_columns = pd.DataFrame(testing_sets[i][0]).columns
        y_test_columns = pd.DataFrame(testing_sets[i][1]).columns
        X_test_inverse.columns = X_test_columns
        y_test_inverse.columns = y_test_columns
        inversed_testing_sets.append((X_test_inverse, y_test_inverse))

    # print(f'\nOdwrocona standaryzacja X: {inversed_testing_sets[6][0]}')
    # print(f'\nOdwrocona standaryzacja y: {inversed_testing_sets[6][1]}')
    return inversed_testing_sets

# Standarized filled training data for neural network
def standarized_filled_testing_sets_to_df2(filled_testing_sets, testing_sets):
    standarized_filled_testing_sets = []

    for i in range(len(filled_testing_sets)):
        X_test, y_test = filled_testing_sets[i]
        X_test_df = pd.DataFrame(X_test, index=testing_sets[i][0].index)
        y_test_df = pd.DataFrame(y_test, index=testing_sets[i][1].index)
        X_test_columns = pd.DataFrame(testing_sets[i][0]).columns
        y_test_columns = pd.DataFrame(testing_sets[i][1]).columns
        X_test_df.columns = X_test_columns
        y_test_df.columns = y_test_columns
        standarized_filled_testing_sets.append((X_test_df, y_test_df))

    # print(f'\nStandaryzowane X jako df: {standarized_filled_testing_sets[6][0]}')
    # print(f'\nStandaryzowane y jako df: {standarized_filled_testing_sets[6][1]}')
    return standarized_filled_testing_sets
