import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.calibration import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer


def remove_sets_with_many_missing_values(original_sets):
    cleaned_sets = []

    for X, y in original_sets:
        missing_values_percentage = (X.isnull().sum().sum() + y.isnull().sum().sum()) / (X.size + y.size)
        if missing_values_percentage <= 0.20:
            cleaned_sets.append((X, y))

    removed_sets = list(set(range(len(original_sets))) - set(range(len(cleaned_sets))))
    # print(f'Removed sets: {removed_sets}')
    return cleaned_sets

def standardize_data(training_sets):
    standardized_training_sets = []
    scalers = []

    for i in range(len(training_sets)):
        X_train, y_train = training_sets[i]
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(training_sets[i][0])
        y_train = y_scaler.fit_transform(training_sets[i][1].values.reshape(-1, 1)).flatten()  # Use the original y_train as the target
        standardized_training_sets.append((X_train, y_train))
        scalers.append((X_scaler, y_scaler))

    # print(f'\nStandaryzowane dane X: {standardized_training_sets[6][0]}') # pierwsza liczba: 0-29 = lata 2016-2017, 30-59 = lata 2016-2018, 60-89 = lata 2016-2019, 90-119 = lata 2016-2020, 120-149 = lata 2016-2021, 150-179 = lata 2016-2022
    # print(f'\nStandaryzowane dane y: {standardized_training_sets[6][1]}') # pierwsza liczba: 0-29 = lata 2016-2017, 30-59 = lata 2016-2018, 60-89 = lata 2016-2019, 90-119 = lata 2016-2020, 120-149 = lata 2016-2021, 150-179 = lata 2016-2022
    return standardized_training_sets, scalers

# dla zbiorow testowych musi byc funckja, ktora uzywa scalerow z danych treningowych
def standardize_testing_sets(testing_sets, scalers):
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

# Fill missing values to find optimal k
def fill_blanks_with_median(standardized_training_sets):
    filled_training_sets_median = []

    for i in range(len(standardized_training_sets)):
        X_train, y_train = standardized_training_sets[i]
        X_train = pd.DataFrame(X_train).fillna(pd.DataFrame(X_train).median()).values
        filled_training_sets_median.append((X_train, y_train))
    # print(f'\nWypelnione mediana X: {filled_training_sets_median[6][0]}')
    # print(f'\nWypelnione mediana y: {filled_training_sets_median[6][1]}')
    return filled_training_sets_median

def find_optimal_k(filled_training_sets_median):
    k_values = range(1, 6)
    optimal_k_values = []

    for _, (X_train, y_train) in enumerate(filled_training_sets_median):
        mse_values_per_set = []
        for k in k_values:
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_train)
            mse = mean_squared_error(y_train, y_pred)
            mse_values_per_set.append(mse)

        optimal_k = k_values[np.argmin(mse_values_per_set)]
        optimal_k_values.append(optimal_k)

    optimal_k = int(np.median(optimal_k_values))
    # plt.plot(range(1, len(filled_training_sets_median) + 1), optimal_k_values)
    # plt.xlabel('Set')
    # plt.ylabel('Optimal k')
    # plt.title('Optimal k for each set')
    # plt.show()
    return optimal_k

# Fill missing values in standardized_training_sets using KNNImputer
def fill_blanks_with_knn(standardized_sets, optimal_k):
    imputer = KNNImputer(n_neighbors=optimal_k)
    filled_sets = []

    for i in range(len(standardized_sets)):
        X, y = standardized_sets[i]
        X_filled = imputer.fit_transform(X)
        y_filled = imputer.fit_transform(y.reshape(-1, 1)).flatten()
        filled_sets.append((X_filled, y_filled))
    # print(f'\nWypelnione przez knn X: {filled_sets[6][0]}')
    # print(f'\nWypelnione przen knn y: {filled_sets[6][1]}')
    return filled_sets

# Perform cross-validation to evaluate the performance of KNN imputation
def cross_validate_filled_sets(filled_sets):
    mse_values = []

    for _, (X, y) in enumerate(filled_sets):
        knn = KNeighborsRegressor(n_neighbors=1)
        y_pred = cross_val_predict(knn, X, y, cv=50)
        mse = mean_squared_error(y, y_pred)
        mse_values.append(mse)

    # Plot the MSE values
    # plt.plot(range(1, len(filled_sets) + 1), mse_values)
    # plt.xlabel('Set')
    # plt.ylabel('Mean Squared Error')
    # plt.title('MSE for each set')
    # plt.show()
    return mse_values

# Inverse standardization of filled_training_sets
def inverse_standardization(filled_sets, original_sets, scalers):
    inverse_filled_sets = []

    for i in range(len(filled_sets)):
        X, y = filled_sets[i]
        X_inverse = scalers[i][0].inverse_transform(X)
        y_inverse = scalers[i][1].inverse_transform(y.reshape(-1, 1)).flatten()
        inverse_filled_sets.append((pd.DataFrame(X_inverse, index=original_sets[i][0].index), pd.DataFrame(y_inverse, index=original_sets[i][1].index)))

    inversed_sets = []

    for i in range(len(inverse_filled_sets)):
        X_inverse, y_inverse = inverse_filled_sets[i]
        X_columns = pd.DataFrame(original_sets[i][0]).columns
        y_columns = pd.DataFrame(original_sets[i][1]).columns
        X_inverse.columns = X_columns
        y_inverse.columns = y_columns
        inversed_sets.append((X_inverse, y_inverse))

    # print(f'\nOdwrocona standaryzacja X: {inversed_sets[160][0]}')
    # print(f'\nOdwrocona standaryzacja y: {inversed_sets[160][1]}')
    return inversed_sets

# Standarized filled training data for neural network
def standarized_filled_sets_to_df(filled_sets, original_sets):
    standarized_filled_sets = []

    for i in range(len(filled_sets)):
        X, y = filled_sets[i]
        X_df = pd.DataFrame(X, index=original_sets[i][0].index)
        y_df = pd.DataFrame(y, index=original_sets[i][1].index)
        X_columns = pd.DataFrame(original_sets[i][0]).columns
        y_columns = pd.DataFrame(original_sets[i][1]).columns
        X_df.columns = X_columns
        y_df.columns = y_columns
        standarized_filled_sets.append((X_df, y_df))

    # print(f'\nStandaryzowane X jako df: {standarized_filled_sets[6][0]}')
    # print(f'\nStandaryzowane y jako df: {standarized_filled_sets[6][1]}')
    return standarized_filled_sets
