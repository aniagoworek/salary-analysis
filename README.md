# **Analysis of the Determinants of Wages in Polish Counties (2016-2022)**

## **Overview**
This project analyzes the determinants of wages in Polish counties from 2016 to 2022 using supervised machine learning methods. The goal is to evaluate the effectiveness of machine learning techniques in forecasting wages and identify key macroeconomic factors influencing wage levels. The analysis uses data from Statistics Poland, including variables such as wage history, demographics, unemployment, transportation infrastructure, and access to social services.

Machine learning models applied include regression trees, bagging, random forests, boosting (XGBoost), and neural networks. The XGBoost model demonstrated exceptional performance, achieving an average relative percentage error of less than 5%.

## **Key Features**
- **Data Handling:**
  - Cleans and preprocesses macroeconomic data for Polish counties.
  - Fills missing data using techniques like median and k-NN imputation.
  - Standardizes datasets to prepare them for machine learning models.

- **Machine Learning Models:**
  - Implements Random Forests, Bagging, Boosting (XGBoost), and Convolutional Neural Networks (CNNs).
  - Hyperparameter tuning to optimize model performance.

- **Feature Importance Analysis:**
  - Identifies key factors influencing wages, such as:
    - Average residential unit size.
    - Age structure of residents.
    - Unemployment levels.
    - Historical wages.

- **Visualization and Diagnostics:**
  - Plots model error distributions and feature importance rankings.
  - Evaluates model performance using Mean Squared Error (MSE) and related metrics.
