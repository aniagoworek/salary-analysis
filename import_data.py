import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def import_data():
    df = pd.read_csv('data.csv', delimiter=';')
    return df

def transform_data(df):
    df['Miejsce'] = df['Miejsce'].astype('category')
    df.drop(['Kod powiatu'], axis=1, inplace=True)
    df.columns = df.columns.str.replace(r'wynagrodzenia', 'salary', regex=True)
    df = df.select_dtypes(include=[np.number])

    print(df)
    print(df.dtypes)

    df_2016_2017 = df.filter(regex='- 201[67]')
    df_2016_2018 = df.filter(regex='- 201[678]')
    df_2016_2019 = df.filter(regex='- 201[6789]')
    df_2016_2020 = df.filter(regex='- 20[12][67890]')
    df_2016_2021 = df.filter(regex='- 20[12][678901]')
    df_2016_2022 = df.filter(regex='- 20[12][6789012]')
    print(df_2016_2020.head())
    return df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022

def split_data(df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022):
    np.random.seed(73)
    training_sets = []
    testing_sets = []

    for df_year in [df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022]:
        X = df_year.drop(columns=[f'salary - {df_year.columns[-1][-4:]}'])
        y = df_year[f'salary - {df_year.columns[-1][-4:]}']

        for _ in range(30): # 30 train and test sets for every year
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            training_sets.append((X_train, y_train))
            testing_sets.append((X_test, y_test))
    print(f'\nInput data X: {testing_sets[2][0]}') # The first number represents the number of the set, the second: 0 - X (training set), 1 - y (testing set)
    print(f'\nInput data y: {training_sets[2][1]}') # The first number: 0-29 = years 2016-2017, 30-59 = years 2016-2018, 60-89 = years 2016-2019, 90-119 = years 2016-2020, 120-149 = years 2016-2021, 150-179 = years 2016-2022
    return testing_sets, training_sets

def split_data_without_salary(df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022):
    np.random.seed(73)
    training_sets = []
    testing_sets = []

    for df_year in [df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022]:
        X = df_year.drop(columns=df_year.filter(regex='salary').columns)
        y = df_year[f'salary - {df_year.columns[-1][-4:]}']

        for _ in range(30):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            training_sets.append((X_train, y_train))
            testing_sets.append((X_test, y_test))
    print(f'\nInput data X: {testing_sets[6][0]}') # The first number represents the number of the set, the second: 0 - X, 1 - y
    print(f'\nInput data y: {training_sets[6][1]}') # The first number: 0-29 = years 2016-2017, 30-59 = years 2016-2018, 60-89 = years 2016-2019, 90-119 = years 2016-2020, 120-149 = years 2016-2021, 150-179 = years 2016-2022
    return testing_sets, training_sets