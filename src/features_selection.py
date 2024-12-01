import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

random_state = 42
test_size = 0.2

df = pd.read_csv('../data/processed/cleaned_heart_attack.csv', sep=',')
columns = df.columns.to_numpy()
target  = "Heart Attack Risk"
df.iloc[:5, 10:20]

x = df.drop(target, axis = 1)
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 42)

#normalisation

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

minmax = MinMaxScaler()
standard = StandardScaler()
robust = RobustScaler()

scaler = minmax
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

##Model-Based Feature Selection

rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
rf.fit(x_train_scaled, y_train)

feature_importance = rf.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': feature_importance
})


feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
top_features = feature_importance_df[feature_importance_df['Importance'] > 0.0098]
top_features

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance using Random Forest')
plt.show()

dff = df.loc[:, top_features.Feature]
dff

# Compute correlation matrix for all variables
corr_matrix = dff.corr()

# Visualize the correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=False, cmap='cool', linewidths=0.1)
plt.title('Correlation Matrix')
plt.show()

dff[target] = y
dff.to_csv('../data/processed/randomForest_selected_features.csv', sep = ',', header = True, index = False)

##Iterative Feature Selection

from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=random_state),
n_features_to_select=15)
select.fit(x_train, y_train)
# visualize the selected features:
mask = select.get_support()
features_rfe = select.get_feature_names_out()
print(features_rfe)

df_rfe= df.loc[:, features_rfe]
df_rfe.loc[:, target] = y
corr_matrix_rfe = df_rfe.corr()
# Visualize the correlation matrix using a heatmap
sns.heatmap(corr_matrix_rfe, annot=False, cmap='cool', linewidths=0.1)
plt.title('Correlation Matrix')
plt.show()
df_rfe[target].value_counts(), y.value_counts()

df_rfe.to_csv('../data/processed/RFE_selected_features.csv', sep = ',', header = True, index = False)