from transform_data import import_data, transform_data, split_data
from transform_training_sets import remove_sets_with_many_missing_values, standardize_data, fill_blanks_with_median, find_optimal_k, fill_blanks_with_knn, cross_validate_filled_sets, inverse_standardization, standarized_filled_training_sets_to_df
from transform_test_sets import remove_sets_with_many_missing_values2, standardize_data2, fill_blanks_with_knn2, cross_validate_filled_sets2, inverse_standardization2, standarized_filled_testing_sets_to_df2
from models import model_rf, model_boost, model_boost, model_bagging, model_cnn, tune_bagging_hyperparameters, find_optimal_hyperparameters

if __name__ == '__main__':
    df = import_data()
    df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022 = transform_data(df)
    testing_sets, training_sets = split_data(df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022)
    
    training_sets = remove_sets_with_many_missing_values(training_sets)
    standardized_training_sets, scalers = standardize_data(training_sets)
    training_sets_filled = fill_blanks_with_median(standardized_training_sets)
    optimal_k = find_optimal_k(training_sets_filled)
    training_sets_filled_knn = fill_blanks_with_knn(training_sets_filled, optimal_k)
    mse_values_trainig_set = cross_validate_filled_sets(training_sets_filled_knn)
    inversed_filled_training_sets = inverse_standardization(training_sets_filled_knn, training_sets, scalers) # do lasow
    standarized_filled_training_sets = standarized_filled_training_sets_to_df(training_sets_filled, training_sets) # do sieci neuronowych

    testing_sets = remove_sets_with_many_missing_values2(testing_sets)
    standardized_testing_sets = standardize_data2(testing_sets, scalers)
    testing_sets_filled = fill_blanks_with_knn2(standardized_testing_sets, optimal_k)
    mse_values_testing_set = cross_validate_filled_sets2(testing_sets_filled)
    inversed_filled_testing_sets = inverse_standardization2(testing_sets_filled, testing_sets, scalers)
    standarized_filled_testing_sets = standarized_filled_testing_sets_to_df2(testing_sets_filled, testing_sets)

    par_for_models_2016_2017 = tune_bagging_hyperparameters(inversed_filled_training_sets, k = 0, n = 3)
    par_for_models_2016_2018 = tune_bagging_hyperparameters(inversed_filled_training_sets, k = 4, n = 6)
    par_for_models_2016_2019 = tune_bagging_hyperparameters(inversed_filled_training_sets, k = 7, n = 9)
    par_for_models_2016_2020 = tune_bagging_hyperparameters(inversed_filled_training_sets, k = 10, n = 12)
    par_for_models_2016_2021 = tune_bagging_hyperparameters(inversed_filled_training_sets, k = 13, n = 15)
    par_for_models_2016_2022 = tune_bagging_hyperparameters(inversed_filled_training_sets, k = 16, n = 18)
    
    par_for_boost_2016_2017 = find_optimal_hyperparameters(inversed_filled_training_sets, inversed_filled_testing_sets, k = 0, n = 29)
    par_for_boost_2016_2018 = find_optimal_hyperparameters(inversed_filled_training_sets, inversed_filled_testing_sets, k = 30, n = 59)
    par_for_boost_2016_2019 = find_optimal_hyperparameters(inversed_filled_training_sets, inversed_filled_testing_sets, k = 60, n = 89)
    par_for_boost_2016_2020 = find_optimal_hyperparameters(inversed_filled_training_sets, inversed_filled_testing_sets, k = 90, n = 119)
    par_for_boost_2016_2021 = find_optimal_hyperparameters(inversed_filled_training_sets, inversed_filled_testing_sets, k = 120, n = 149)
    par_for_boost_2016_2022 = find_optimal_hyperparameters(inversed_filled_training_sets, inversed_filled_testing_sets, k = 150, n = 179)
    
    rf_2016_2017 = model_rf(inversed_filled_training_sets, inversed_filled_testing_sets, k = 0, n = 29, n_estimators = 200, max_depth = 10, min_samples_split = 5)
    rf_2016_2018 = model_rf(inversed_filled_training_sets, inversed_filled_testing_sets, k = 30, n = 59, n_estimators = 200, max_depth = None, min_samples_split = 2)
    rf_2016_2019 = model_rf(inversed_filled_training_sets, inversed_filled_testing_sets, k = 60, n = 89, n_estimators = 50, max_depth = 5, min_samples_split = 10)
    rf_2016_2020 = model_rf(inversed_filled_training_sets, inversed_filled_testing_sets, k = 90, n = 119, n_estimators = 100, max_depth = 10, min_samples_split = 5)
    rf_2016_2021 = model_rf(inversed_filled_training_sets, inversed_filled_testing_sets, k = 120, n = 149, n_estimators = 50, max_depth = 10, min_samples_split = 2)
    rf_2016_2022 = model_rf(inversed_filled_training_sets, inversed_filled_testing_sets, k = 150, n = 179, n_estimators = 200, max_depth = None, min_samples_split = 5)

    boost_2016_2017 = model_boost(inversed_filled_training_sets, inversed_filled_testing_sets, k = 0, n = 29)
    boost_2016_2018 = model_boost(inversed_filled_training_sets, inversed_filled_testing_sets, k = 30, n = 59)
    boost_2016_2019 = model_boost(inversed_filled_training_sets, inversed_filled_testing_sets, k = 60, n = 89)
    boost_2016_2020 = model_boost(inversed_filled_training_sets, inversed_filled_testing_sets, k = 90, n = 119)
    boost_2016_2021 = model_boost(inversed_filled_training_sets, inversed_filled_testing_sets, k = 120, n = 149)
    boost_2016_2022 = model_boost(inversed_filled_training_sets, inversed_filled_testing_sets, k = 150, n = 179)

    bagging_2016_2017 = model_bagging(inversed_filled_training_sets, inversed_filled_testing_sets, k = 0, n = 29, bootstrap = True, bootstrap_features = False, max_features = 1.0, max_samples = 1.0, n_estimators = 50)
    bagging_2016_2018 = model_bagging(inversed_filled_training_sets, inversed_filled_testing_sets, k = 30, n = 59, bootstrap = False, bootstrap_features = False, max_features = 1.0, max_samples = 0.7, n_estimators = 50)
    bagging_2016_2019 = model_bagging(inversed_filled_training_sets, inversed_filled_testing_sets, k = 60, n = 89, bootstrap = False, bootstrap_features = False, max_features = 0.7, max_samples = 0.7, n_estimators = 100)
    bagging_2016_2020 = model_bagging(inversed_filled_training_sets, inversed_filled_testing_sets, k = 90, n = 119, bootstrap = True, bootstrap_features = False, max_features = 1.0, max_samples = 1.0, n_estimators = 100)
    bagging_2016_2021 = model_bagging(inversed_filled_training_sets, inversed_filled_testing_sets, k = 120, n = 149, bootstrap = True, bootstrap_features = False, max_features = 1.0, max_samples = 1.0, n_estimators = 50)
    bagging_2016_2022 = model_bagging(inversed_filled_training_sets, inversed_filled_testing_sets, k = 150, n = 179, bootstrap = False, bootstrap_features = False, max_features = 0.7, max_samples = 0.5, n_estimators = 100)

    cnn_2016_2017 = model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k = 0, n = 29)
    cnn_2016_2018 = model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k = 30, n = 59)
    cnn_2016_2019 = model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k = 60, n = 89)
    cnn_2016_2020 = model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k = 90, n = 119)
    cnn_2016_2021 = model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k = 120, n = 149)
    cnn_2016_2022 = model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k = 150, n = 179)
