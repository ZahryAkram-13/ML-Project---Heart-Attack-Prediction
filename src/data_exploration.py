
#Import necessary library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder
#Installer category_encoders
!pip install category_encoders
from category_encoders.target_encoder import TargetEncoder


#Load the dataset 'heart_attack_prediction_dataset.csv' into a DataFrame named df_Heart
df_Heart = pd.read_csv('heart_attack_prediction_dataset.csv')
df_Heart.head()

#Displays summary information about the DataFrame,
df_Heart.info()

#Generates summary statistics for numeric columns and transposes the output for easier reading.
df_Heart.describe().T

"""La colonne **"Blood Pressure"** est mise à jour dans le DataFrame. Chaque valeur est transformée en un entier basé sur la formule :

$$
\text{Nouvelle valeur} = 2 \times \text{diastolique} + \frac{\text{systolique}}{3}
$$


"""

#Converts 'Blood Pressure' values from string format "systolic/diastolic" to a numerical value based on a formula.
df_Heart['Blood Pressure'] = df_Heart['Blood Pressure'].apply(lambda x:int(2 * int(x.split("/")[1]) + int(x.split("/")[0]) / 3))
df_Heart['Blood Pressure'].head()

#Creates histograms for all numeric columns in the DataFrame with 50 bins and a large figure size for better visualization.
df_Heart.hist(bins=50, figsize=(50,35))
plt.show()

# Converts all columns with object data types to the 'string' data type for consistency.
string_col = df_Heart.select_dtypes(include="object").columns
df_Heart[string_col]=df_Heart[string_col].astype("string")

#Defines a list of categorical feature columns in the dataset for further analysis or encoding.
categorical_features = ['Sex', 'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Diet',
        'Previous Heart Problems', 'Medication Use', 'Stress Level', 'Physical Activity Days Per Week',
        'Sleep Hours Per Day', 'Continent', 'Hemisphere', 'Country']

#Plots bar charts and pie charts for each categorical feature to visualize the distribution of values.
plt.figure(figsize=(12,65))

i = 0
j = 0

for col in categorical_features:
    feature = df_Heart.groupby(col)[col].count() #Counts occurrences of each category in the column
    plt.subplot(15, 2, i+1) #Creates a subplot for the bar chart
    plt.bar(x=feature.index, height=feature.values)
    plt.title(col, fontsize=15)
    plt.xlabel("")

    if col == "Country":
        plt.xticks(rotation=90) #Rotates x-ticks for better readability if the column is "Country"
    if col == "Continent":
        plt.xticks(rotation=10) #Slightly rotates x-ticks for "Continent"

    plt.subplot(15, 2, j+2) #Creates a subplot for the pie chart
    plt.pie(x=feature.values, autopct="%.1f%%", pctdistance=0.8, labels=feature.index)
    plt.title(col, fontsize=15)
    i += 2
    j += 2
plt.show()

#Displays the data types of all columns in the DataFrame `df_Heart`.
df_Heart.dtypes

#Separates the columns into two lists: one for string columns and one for numeric columns.
string_col = df_Heart.select_dtypes("string").columns.to_list()
num_col = df_Heart.columns.to_list()
for col in string_col:
    num_col.remove(col) #Removes string columns from the list of all columns, leaving only numeric columns

#Generates summary statistics for numeric columns and transposes the output for easier reading.
df_Heart.describe().T

#Creates a new DataFrame `df_only_numbers` containing only the numeric columns from `df_Heart`.
df_only_numbers = df_Heart[num_col]
#df_only_numbers.columns

#Prepares the data by encoding categorical variables, creating dummy variables, and selecting relevant columns for analysis.

data = df_Heart.copy() #Creates a copy of the original DataFrame to avoid modifying the original data.

#Label encoding for categorical columns 'Sex' and 'Hemisphere'
le = LabelEncoder()
target_encoder = TargetEncoder()
data[["Sex", "Hemisphere"]] = data[["Sex", "Hemisphere"]].agg(le.fit_transform)

#Target encoding for 'Continent' and 'Country' based on 'Heart Attack Risk' as the target variable
data["Continent"] = target_encoder.fit_transform(data["Continent"], df_Heart["Heart Attack Risk"])
data["Country"] = target_encoder.fit_transform(data["Country"], df_Heart["Heart Attack Risk"])

#Creates dummy variables for the 'Diet' column
data = pd.get_dummies(data=data, columns=["Diet"])

#Selects a subset of columns that are relevant for the analysis
data = data[['Sex', 'Heart Attack Risk', 'Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week'
        , 'Sedentary Hours Per Day', 'Income', 'BMI',
        'Triglycerides', 'Stress Level', 'Physical Activity Days Per Week', 'Sleep Hours Per Day',
        'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption','Previous Heart Problems',
        'Medication Use', 'Country', 'Continent', 'Hemisphere']]
data.head()

#Calculates and visualizes the correlation of variables with 'Heart Attack Risk' using a bar plot.
heart_attack_corr = data.corr()['Heart Attack Risk'] #Calculates the correlation of all features with 'Heart Attack Risk'
heart_attack_corr = heart_attack_corr.drop("Heart Attack Risk", axis=0).sort_values(ascending=False) #Drops 'Heart Attack Risk' and sorts correlations in descending orde

#Creates a bar plot to visualize the correlation with 'Heart Attack Risk'
plt.figure(figsize=(10,5))
sns.set(font_scale=0.8)
sns.barplot(x=heart_attack_corr.index, y=heart_attack_corr, color="#D1A0D1")

# Customizing the plot
plt.xticks(rotation=90)
plt.ylim(-0.02, 0.05)
plt.title("Relation des variables avec Heart Attack Risk", fontsize=15)
plt.show()

#Computes and visualizes the correlation matrix of all variables in the data using a heatmap.
correlation_matrix = data.corr()

#Creates a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix", fontsize=15)
plt.show()

#Calculates and visualizes the percentage of smokers by sex using a stacked bar chart.
smoking_percentage = df_Heart.groupby('Sex')['Smoking'].value_counts(normalize=True).unstack().fillna(0) * 100

#Creates a stacked bar chart to visualize the smoking percentages by sex
smoking_percentage.plot(kind='bar', figsize=(8, 6), color=['skyblue', 'lightcoral'])

plt.title("Percentage of Smoking by Sex", fontsize=15)
plt.xlabel("Sex", fontsize=12)
plt.ylabel("Percentage of Smoking (%)", fontsize=12)


plt.xticks(rotation=0)
plt.legend(title="Smoking", labels=["Non-smoker", "Smoker"])
plt.show()

#Converts specified columns to numeric values and creates boxplots for 'Blood Pressure' and 'Cholesterol'.
columns_specify = ['Blood Pressure', 'Cholesterol']
df_Heart[columns_specify] = df_Heart[columns_specify].apply(pd.to_numeric, errors='coerce')

#Creates a boxplot to visualize the distribution of the specified columns
plt.figure(figsize=(20, 10))
plt.boxplot(df_Heart[columns_specify].values, labels=columns_specify)
plt.title('Boxplots of Multiple Variables')
plt.xlabel('Variables')
plt.ylabel('Values')
plt.show()

#Converts specified columns to numeric values and creates boxplots for 'Exercise Hours Per Week', 'Sedentary Hours Per Day', and 'BMI'.
columns_specify1 = ['Exercise Hours Per Week', 'Sedentary Hours Per Day', 'BMI']
df_Heart[columns_specify1] = df_Heart[columns_specify1].apply(pd.to_numeric, errors='coerce')

#Creates a boxplot to visualize the distribution of the specified columns
plt.figure(figsize=(10, 7))
plt.boxplot(df_Heart[columns_specify1].values, labels=columns_specify1)
plt.title('Boxplots of Multiple Variables')
plt.xlabel('Variables')
plt.ylabel('Values')
plt.show()

