from import_data import import_data, transform_data, split_data_without_salary
from transform_data_sets import standardize_data, standardize_testing_sets, fill_blanks_with_median, find_optimal_k, fill_blanks_with_knn, cross_validate_filled_sets, inverse_standardization, standarized_filled_sets_to_df
from tune_models_parameters import tune_rf_parameters, tune_bagging_parameters, tune_boosting_parameters, tune_cnn_parameters, tune_xgboost_parameters, helper_function, Pool, partial
from models import model_boost, model_rf, model_bagging, model_cnn, model_xgboost
from visualization import plot_all_errors, plot_feature_importances, plot_all_r2


if __name__ == '__main__':
    df = import_data()
    df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022 = transform_data(df)
    testing_sets, training_sets = split_data_without_salary(df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022)
    
    standardized_training_sets, scalers = standardize_data(training_sets)
    training_sets_filled = fill_blanks_with_median(standardized_training_sets)
    optimal_k = find_optimal_k(training_sets_filled)
    training_sets_filled_knn = fill_blanks_with_knn(training_sets_filled, optimal_k)
    mse_values_training_set = cross_validate_filled_sets(training_sets_filled_knn)
    final_training_sets = inverse_standardization(training_sets_filled_knn, training_sets, scalers) # do lasow
    standarized_filled_training_sets = standarized_filled_sets_to_df(training_sets_filled, training_sets) # do sieci neuronowych

    standardized_testing_sets = standardize_testing_sets(testing_sets, scalers)
    testing_sets_filled = fill_blanks_with_knn(standardized_testing_sets, optimal_k)
    mse_values_testing_set = cross_validate_filled_sets(testing_sets_filled)
    final_testing_sets = inverse_standardization(testing_sets_filled, testing_sets, scalers)
    standarized_filled_testing_sets = standarized_filled_sets_to_df(testing_sets_filled, testing_sets)

    # with Pool(processes=8) as pool:
    #     partial_function = partial(helper_function, final_training_sets)
    #     par_for_boosting = pool.map(partial_function, range(15, 29))

    # with Pool(processes=4) as pool:
    #     partial_function = partial(helper_function, final_training_sets)
    #     par_for_xgb = pool.map(partial_function, range(0, 179, 30))    

    # for k in range(23, 30):
    #     par_for_rf = tune_rf_parameters(final_training_sets, k)

    # for k in range(0, 30):
    #     par_for_bagging = tune_bagging_parameters(final_training_sets, k)

    # for k in range(0, 30):
    #     par_for_boosting = tune_boosting_parameters(final_training_sets, k)

    # for k in range(0, 180, 30):
    #     par_for_cnn = tune_cnn_parameters(final_training_sets, k, k+29)

    # boost_2016_2017, boost_errors_2016_2017, boost_feature_2016_2017, variables_names_2016_2017 = model_boost(final_training_sets, final_testing_sets, k = 0, n = 4, max_depth = 5, max_features = 1.0, min_samples_leaf = 2, min_samples_split = 2, n_estimators = 20)
    # boost_2016_2018, boost_errors_2016_2018, boost_feature_2016_2018, variables_names_2016_2018 = model_boost(final_training_sets, final_testing_sets, k = 5, n = 9, max_depth = 5, max_features = 1.0, min_samples_leaf = 2, min_samples_split = 15, n_estimators = 6)
    # boost_2016_2019, boost_errors_2016_2019, boost_feature_2016_2019, variables_names_2016_2019 = model_boost(final_training_sets, final_testing_sets, k = 10, n = 14, max_depth = 5, max_features = 1.0, min_samples_leaf = 1, min_samples_split = 15, n_estimators = 180)
    # boost_2016_2020, boost_errors_2016_2020 = model_boost(final_training_sets, final_testing_sets, k = 90, n = 119, max_depth = 5, max_features = 1.0, min_samples_leaf = 1, min_samples_split = 15, n_estimators = 270)
    # boost_2016_2021, boost_errors_2016_2021 = model_boost(final_training_sets, final_testing_sets, k = 120, n = 149, max_depth = 10, max_features = 1.0, min_samples_leaf = 1, min_samples_split = 15, n_estimators = 150)
    # boost_2016_2022, boost_errors_2016_2022 = model_boost(final_training_sets, final_testing_sets, k = 150, n = 179, max_depth = 20, max_features = 1.0, min_samples_leaf = 2, min_samples_split = 15, n_estimators = 210)

    # rf_2016_2017, rf_errors_2016_2017 = model_rf(final_training_sets, final_testing_sets, k = 0, n = 29, n_estimators = 50, max_depth = 5, min_samples_split = 15)
    # rf_2016_2018, rf_errors_2016_2018 = model_rf(final_training_sets, final_testing_sets, k = 30, n = 59, n_estimators = 30, max_depth = 10, min_samples_split = 2)
    # rf_2016_2019, rf_errors_2016_2019 = model_rf(final_training_sets, final_testing_sets, k = 60, n = 89, n_estimators = 50, max_depth = 5, min_samples_split = 2)
    # rf_2016_2020, rf_errors_2016_2020 = model_rf(final_training_sets, final_testing_sets, k = 90, n = 119, n_estimators = 100, max_depth = 15, min_samples_split = 5)
    # rf_2016_2021, rf_errors_2016_2021 = model_rf(final_training_sets, final_testing_sets, k = 120, n = 149, n_estimators = 100, max_depth = None, min_samples_split = 15)
    # rf_2016_2022, rf_errors_2016_2022 = model_rf(final_training_sets, final_testing_sets, k = 150, n = 179, n_estimators = 200, max_depth = None, min_samples_split = 15)

    # bagging_2016_2017, bagging_errors_2016_2017 = model_bagging(final_training_sets, final_testing_sets, k = 0, n = 29, bootstrap = True, bootstrap_features = True, max_features = 0.5, max_samples = 0.5, n_estimators = 100)
    # bagging_2016_2018, bagging_errors_2016_2018 = model_bagging(final_training_sets, final_testing_sets, k = 30, n = 59, bootstrap = True, bootstrap_features = True, max_features = 0.5, max_samples = 1.0, n_estimators = 30)
    # bagging_2016_2019, bagging_errors_2016_2019 = model_bagging(final_training_sets, final_testing_sets, k = 60, n = 89, bootstrap = True, bootstrap_features = True, max_features = 0.5, max_samples = 0.5, n_estimators = 100)
    # bagging_2016_2020, bagging_errors_2016_2020 = model_bagging(final_training_sets, final_testing_sets, k = 90, n = 119, bootstrap = True, bootstrap_features = True, max_features = 0.5, max_samples = 0.7, n_estimators = 50)
    # bagging_2016_2021, bagging_errors_2016_2021 = model_bagging(final_training_sets, final_testing_sets, k = 120, n = 149, bootstrap = True, bootstrap_features = False, max_features = 0.7, max_samples = 0.5, n_estimators = 200)
    # bagging_2016_2022, bagging_errors_2016_2022 = model_bagging(final_training_sets, final_testing_sets, k = 150, n = 179, bootstrap = False, bootstrap_features = False, max_features = 0.7, max_samples = 0.5, n_estimators = 100)

    # cnn_2016_2017, cnn_errors_2016_2017 = model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k = 0, n = 29)
    # cnn_2016_2018, cnn_errors_2016_2017 = model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k = 30, n = 59)
    # cnn_2016_2019, cnn_errors_2016_2017 = model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k = 60, n = 89)
    # cnn_2016_2020, cnn_errors_2016_2017 = model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k = 90, n = 119)
    # cnn_2016_2021, cnn_errors_2016_2017 = model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k = 120, n = 149)
    # cnn_2016_2022, cnn_errors_2016_2017 = model_cnn(standarized_filled_training_sets, standarized_filled_testing_sets, k = 150, n = 179)

    xgb_2016_2017, xgb_errors_2016_2017, xgb_feature_2016_2017, variables_names_2016_2017, r2_2016_2017 = model_xgboost(final_training_sets, final_testing_sets, k = 0, n = 29, max_depth = 7, learning_rate = 0.01, subsample = 0.8, colsample_bytree = 0.8, n_estimators = 200)
    xgb_2016_2018, xgb_errors_2016_2018, xgb_feature_2016_2018, variables_names_2016_2018, r2_2016_2018 = model_xgboost(final_training_sets, final_testing_sets, k = 30, n = 59, max_depth = 7, learning_rate = 0.01, subsample = 0.8, colsample_bytree = 0.8, n_estimators = 300)
    xgb_2016_2019, xgb_errors_2016_2019, xgb_feature_2016_2019, variables_names_2016_2019, r2_2016_2019 = model_xgboost(final_training_sets, final_testing_sets, k = 60, n = 89, max_depth = 3, learning_rate = 0.2, subsample = 0.9, colsample_bytree = 0.8, n_estimators = 30)
    xgb_2016_2020, xgb_errors_2016_2020, xgb_feature_2016_2020, variables_names_2016_2020, r2_2016_2020 = model_xgboost(final_training_sets, final_testing_sets, k = 90, n = 119, max_depth = 3, learning_rate = 0.1, subsample = 0.9, colsample_bytree = 0.9, n_estimators = 30)
    xgb_2016_2021, xgb_errors_2016_2021, xgb_feature_2016_2021, variables_names_2016_2021, r2_2016_2021 = model_xgboost(final_training_sets, final_testing_sets, k = 120, n = 149, max_depth = 3, learning_rate = 0.2, subsample = 0.9, colsample_bytree = 0.8, n_estimators = 30)
    xgb_2016_2022, xgb_errors_2016_2022, xgb_feature_2016_2022, variables_names_2016_2022, r2_2016_2022 = model_xgboost(final_training_sets, final_testing_sets, k = 150, n = 179, max_depth = 3, learning_rate = 0.01, subsample = 1.0, colsample_bytree = 1.0, n_estimators = 400)

    boxplots = plot_all_errors(xgb_errors_2016_2017, xgb_errors_2016_2018, xgb_errors_2016_2019, xgb_errors_2016_2020, xgb_errors_2016_2021, xgb_errors_2016_2022)
    
    # r2 = plot_all_r2(r2_2016_2017, r2_2016_2018, r2_2016_2019, r2_2016_2020, r2_2016_2021, r2_2016_2022)

    for i in range(2017, 2023):
        features = plot_feature_importances(globals()[f"variables_names_2016_{i}"], globals()[f"xgb_feature_2016_{i}"])
