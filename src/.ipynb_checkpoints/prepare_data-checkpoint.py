"""
in this python file we tried to prepare our data and do some transformation and operations on diffrent kind of columns.

our goal is to clean our data from any missing values and fill it up and transform any non numerical value to numirical.
 
1/split composing columns to two diffrent columns
2/change sex columns from object to 1/0 values
3/change categorical ordinal columns using OrdinalEncoder
4/change categorical non ordinal columns using OnehotEncoder
5/drop all non numirical columns from data frame
6/ save our cleaned data in a new file cleaned_heart_attack.csv 

"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

random_state = 42

df = pd.read_csv('dataset.csv', sep=',')

sex = df.Sex
binary_sex = sex.apply(lambda x: 1 if x == "Male" else 0)

df['Binary_sex'] = binary_sex
df.loc[:, ['Sex', 'Binary_sex']]
df.drop('Sex', axis = 1, inplace = True)

blood_pressure = df['Blood Pressure']
systolic = blood_pressure.apply(lambda x: x.split('/')[0])
diastolic = blood_pressure.apply(lambda x: x.split('/')[1])
df['Systolic'] = systolic
df['Diastolic'] = diastolic
df.loc[:5, ['Systolic', 'Diastolic']]
df.drop('Blood Pressure', axis = 1, inplace = True)


from sklearn.preprocessing import OrdinalEncoder

print(df.Diet.unique())
diet = df.Diet
diet_categories = ['Unhealthy', 'Average', 'Healthy']
ordinal_encoder = OrdinalEncoder(categories = [diet_categories])
encoded_diet = ordinal_encoder.fit_transform(df[['Diet']])
df['Encoded_diet'] = encoded_diet

df.drop('Diet', axis = 1, inplace = True)


from sklearn.preprocessing import OneHotEncoder
country = df.Country
continent = df.Continent
hemisphere = df.Hemisphere

df['Hemisphere'] = hemisphere.apply(lambda x: x.replace('Hemisphere', ''))
print(df['Hemisphere'].value_counts())

qualitative_df = df.loc[:, ['Country', 'Continent', 'Hemisphere']]
print(qualitative_df)
# print(country.unique().size)

onehot_encoder = OneHotEncoder(sparse_output = False)
encoded_qualitative = onehot_encoder.fit_transform(qualitative_df)
encoded_columns = onehot_encoder.get_feature_names_out(qualitative_df.columns)

encoded_df = pd.DataFrame(encoded_qualitative, columns = encoded_columns)

df1 = pd.concat([df, encoded_df], axis = 1)
df1.drop(['Country', 'Continent', 'Hemisphere'], axis = 1, inplace = True)


df1.drop('Patient ID', axis = 1, inplace = True)

df = df1


df.to_csv('cleaned_heart_attack.csv', sep = ',', header = True, index = False)






