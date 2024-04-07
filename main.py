from transform_data import import_data, transform_data, split_data
from transform_training_sets import remove_sets_with_many_missing_values, standardize_data, fill_blanks_with_median, find_optimal_k, fill_blanks_with_knn, cross_validate_filled_sets, inverse_standardization, standarized_filled_training_sets_to_df
# from transform_test_sets import

if __name__ == '__main__':
    df = import_data()
    df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022 = transform_data(df)
    testing_sets, training_sets = split_data(df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022)
    training_sets = remove_sets_with_many_missing_values(training_sets)

    standardized_training_sets, scalers = standardize_data(training_sets)
    training_sets_filled = fill_blanks_with_median(standardized_training_sets)
    optimal_k = find_optimal_k(training_sets_filled)
    training_sets_filled_knn = fill_blanks_with_knn(training_sets_filled, optimal_k)
    mse_values = cross_validate_filled_sets(training_sets_filled_knn)
    inversed_filled_training_sets = inverse_standardization(training_sets_filled_knn, training_sets, scalers) # do lasow
    standarize_filled_training_sets = standarized_filled_training_sets_to_df(training_sets_filled, training_sets) # do sieci neuronowych
