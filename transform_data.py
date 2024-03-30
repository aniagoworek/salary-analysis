import pandas as pd
import numpy as np

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

df_transposed.insert(0, 'Variables', df_transposed.index)
df_transposed.reset_index(drop=True, inplace=True)
df_transposed = df_transposed.iloc[1:] # delete unneccesary row
df_transposed.reset_index(drop=True, inplace=True)
print(df_transposed) 

# conversion of regions into indexes
df_transposed.columns = df_transposed.iloc[0] #first row as the column header
df_transposed.index = df_transposed.index.rename('Variables') #first column as a side index
df_transposed = df_transposed.iloc[1:] #delete the first row that has become the column header
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
print(df_numeric)

print("Data types:")
print(df_numeric.dtypes)

# add y
y = df_numeric.loc[df_numeric.iloc[:, 0] == "wynagrodzenia"]
print(y)

# add X
df_without_y = df_numeric[df_numeric['Variable'] != 'wynagrodzenia']
X = df_without_y.sort_values(by=['Variable', 'Year'])
print(X)