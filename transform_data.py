import pandas as pd
import numpy as np
import pickle
import os

# Import data
with open('data.csv', 'r') as file:
    lines = file.readlines()

# Data as list
data = [line.strip().split(';') for line in lines]

variables = data[0]
dates_row = data[1][2:]
dates = pd.to_datetime(dates_row, format='%Y')
df = pd.DataFrame(data[1:], columns=variables)
df_transposed = df.transpose()
# print(df)

df_transposed.insert(0, 'Variables', df_transposed.index)
df_transposed.reset_index(drop=True, inplace=True)
df_transposed = df_transposed.iloc[1:] # delete unneccesary row
df_transposed.reset_index(drop=True, inplace=True)
# print(df_transposed) 

# conversion of regions into indexes
df_transposed.columns = df_transposed.iloc[0] #first row as the column header
df_transposed = df_transposed.iloc[1:] # delete the first row that has become the column header
# print(df_transposed)

df_transposed.rename(columns={'Nazwa': 'Variable'}, inplace=True)
df_transposed.rename(columns={'': 'Year'}, inplace=True)
# print(df_transposed)

# convert data to float
def convert_to_numeric(value):
    try:
        return pd.to_numeric(value)
    except ValueError:
        return value


df_numeric = df_transposed.applymap(convert_to_numeric)
df_numeric['Variable'] = df_numeric['Variable'].astype('category')
df_numeric['Year'] = pd.to_datetime(df_numeric['Year'], format='%Y')
# print(df_numeric)
# print("Data types:")
# print(df_numeric.dtypes)

# save places, years and variables
places = df_numeric.index.unique(level=0)
years = df_numeric['Year']
variables = df_numeric['Variable']
print(df_numeric)

# add y
y = df_numeric.loc[df_numeric.iloc[:, 0] == "wynagrodzenia"]
# print(y)
# print(y.drop('Year', axis=1).describe())

# add X
df_without_y = df_numeric[df_numeric['Variable'] != 'wynagrodzenia']
X = df_without_y.sort_values(by='Variable')
# print(X)

# create random division of y and X into training and test sets
np.random.seed(73)
num_sets = 30 # number of training and test sets

total_columns = X.shape[1] - 2  # number of columns in X without column of variables and dates
train_size = int(0.75 * total_columns)

train_indices_list = []
test_indices_list = []

for _ in range(num_sets):
    train_indices = np.random.choice(total_columns, train_size, replace=True)
    test_indices = np.setdiff1d(np.arange(total_columns), train_indices)
    
    train_indices_list.append(train_indices)
    test_indices_list.append(test_indices)

# create 30 training and test sets for y and X
train_y_list = [y.iloc[:, train_indices] for train_indices in train_indices_list]
test_y_list = [y.iloc[:, test_indices] for test_indices in test_indices_list]
train_X_list = [X.iloc[:, train_indices] for train_indices in train_indices_list]
test_X_list = [X.iloc[:, test_indices] for test_indices in test_indices_list]
# print(train_X_list[9])
# print(train_y_list[0])

# save training and test sets into pickle files
folder_path = "training_and_test_sets"
os.makedirs(folder_path, exist_ok=True)
datasets = [train_y_list, test_y_list, train_X_list, test_X_list]

for name, dataset_list in zip(["train_y", "test_y", "train_X", "test_X"], datasets):
    for i, dataset_df in enumerate(dataset_list):
        with open(os.path.join(folder_path, f"{name}_{i+1}.pkl"), "wb") as f:
            pickle.dump(dataset_df, f)