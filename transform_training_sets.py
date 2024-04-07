import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.calibration import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer


def remove_sets_with_many_missing_values(training_sets):
    cleaned_training_sets = []

    for X_train, y_train in training_sets:
        missing_values_percentage = (X_train.isnull().sum().sum() + y_train.isnull().sum().sum()) / (X_train.size + y_train.size)
        if missing_values_percentage <= 0.20:
            cleaned_training_sets.append((X_train, y_train))

    removed_sets = list(set(range(len(training_sets))) - set(range(len(cleaned_training_sets))))
    print(f'Removed sets: {removed_sets}')
    return cleaned_training_sets

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
def fill_blanks_with_knn(standardized_training_sets, optimal_k):
    imputer = KNNImputer(n_neighbors=optimal_k)
    filled_training_sets = []

    for i in range(len(standardized_training_sets)):
        X_train, y_train = standardized_training_sets[i]
        X_train_filled = imputer.fit_transform(X_train)
        y_train_filled = imputer.fit_transform(y_train.reshape(-1, 1)).flatten()
        filled_training_sets.append((X_train_filled, y_train_filled))
    # print(f'\nWypelnione przez knn X: {filled_training_sets[6][0]}')
    # print(f'\nWypelnione przen knn y: {filled_training_sets[6][1]}')
    return filled_training_sets

# Perform cross-validation to evaluate the performance of KNN imputation
def cross_validate_filled_sets(filled_training_sets):
    mse_values = []

    for _, (X_train, y_train) in enumerate(filled_training_sets):
        knn = KNeighborsRegressor(n_neighbors=1)
        y_pred = cross_val_predict(knn, X_train, y_train, cv=50)
        mse = mean_squared_error(y_train, y_pred)
        mse_values.append(mse)

    # Plot the MSE values
    # plt.plot(range(1, len(filled_training_sets) + 1), mse_values)
    # plt.xlabel('Set')
    # plt.ylabel('Mean Squared Error')
    # plt.title('MSE for each set')
    # plt.show()
    return mse_values

# Inverse standardization of filled_training_sets
def inverse_standardization(filled_training_sets, training_sets, scalers):
    inverse_filled_training_sets = []

    for i in range(len(filled_training_sets)):
        X_train, y_train = filled_training_sets[i]
        X_train_inverse = scalers[i][0].inverse_transform(X_train)
        y_train_inverse = scalers[i][1].inverse_transform(y_train.reshape(-1, 1)).flatten()
        inverse_filled_training_sets.append((pd.DataFrame(X_train_inverse, index=training_sets[i][0].index), pd.DataFrame(y_train_inverse, index=training_sets[i][1].index)))

    inversed_training_sets = []

    for i in range(len(inverse_filled_training_sets)):
        X_train_inverse, y_train_inverse = inverse_filled_training_sets[i]
        X_train_columns = pd.DataFrame(training_sets[i][0]).columns
        y_train_columns = pd.DataFrame(training_sets[i][1]).columns
        X_train_inverse.columns = X_train_columns
        y_train_inverse.columns = y_train_columns
        inversed_training_sets.append((X_train_inverse, y_train_inverse))

    print(f'\nOdwrocona standaryzacja X: {inversed_training_sets[6][0]}')
    print(f'\nOdwrocona standaryzacja y: {inversed_training_sets[6][1]}')
    return inversed_training_sets

# Standarized filled training data for neural network
def standarized_filled_training_sets_to_df(filled_training_sets, training_sets):
    standarized_filled_training_sets = []

    for i in range(len(filled_training_sets)):
        X_train, y_train = filled_training_sets[i]
        X_train_df = pd.DataFrame(X_train, index=training_sets[i][0].index)
        y_train_df = pd.DataFrame(y_train, index=training_sets[i][1].index)
        X_train_columns = pd.DataFrame(training_sets[i][0]).columns
        y_train_columns = pd.DataFrame(training_sets[i][1]).columns
        X_train_df.columns = X_train_columns
        y_train_df.columns = y_train_columns
        standarized_filled_training_sets.append((X_train_df, y_train_df))

    # print(f'\nStandaryzowane X jako df: {standarized_filled_training_sets[6][0]}')
    # print(f'\nStandaryzowane y jako df: {standarized_filled_training_sets[6][1]}')
    return standarized_filled_training_sets
