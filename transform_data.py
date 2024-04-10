import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def import_data():
    df = pd.read_csv('data.csv', delimiter=';')
    return df

def transform_data(df):
    df['Miejsce'] = df['Miejsce'].astype('category')
    df.drop(['Kod powiatu'], axis=1, inplace=True)
    df = df.select_dtypes(include=[np.number])

    # print(df)
    # print(df.dtypes)

    df_2016_2017 = df.filter(regex='- 201[67]')
    df_2016_2018 = df.filter(regex='- 201[678]')
    df_2016_2019 = df.filter(regex='- 201[6789]')
    df_2016_2020 = df.filter(regex='- 201[67890]')
    df_2016_2021 = df.filter(regex='- 201[678901]')
    df_2016_2022 = df.filter(regex='- 201[6789012]')
    # print(df_2016_2017.head())
    return df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022

def split_data(df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022):
    np.random.seed(73)
    training_sets = []
    testing_sets = []

    for df_year in [df_2016_2017, df_2016_2018, df_2016_2019, df_2016_2020, df_2016_2021, df_2016_2022]:
        X = df_year.drop(columns=[f'wynagrodzenia - {df_year.columns[-1][-4:]}'])
        y = df_year[f'wynagrodzenia - {df_year.columns[-1][-4:]}']

        for _ in range(30): # zmienic na 30
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            training_sets.append((X_train, y_train))
            testing_sets.append((X_test, y_test))
    # print(f'\nDane poczatkowe X: {testing_sets[6][0]}') # pierwsza liczba oznacza numer zbioru, druga: 0 - X, 1 - y
    # print(f'\nDane poczatkowe y: {training_sets[6][1]}') # pierwsza liczba: 0-29 = lata 2016-2017, 30-59 = lata 2016-2018, 60-89 = lata 2016-2019, 90-119 = lata 2016-2020, 120-149 = lata 2016-2021, 150-179 = lata 2016-2022
    return testing_sets, training_sets
    # jesli mamy 5 zbiorow z kazdych to: 0-4 = lata 2016-2017, 5-9 = lata 2016-2018, 10-14 = lata 2016-2019, 15-19 = lata 2016-2020, 20-24 = lata 2016-2021, 25-29 = lata 2016-2022