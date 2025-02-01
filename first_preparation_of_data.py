import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

#pip install odfpy

file_path='./manual.ods'
data=pd.read_excel(file_path, engine='odf')

relevant_columns=[
    "ID", "Gender", "Education", "DRUG", "DRUG_0=negative_1=Positive", "Drug_Substance","Smoking", "Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)",
    "SKID_Diagnoses", "SKID_Diagnoses 1", "SKID_Diagnoses 2", "Comments_SKID_assessment", "Alcohol_Dependence_In_1st-3rd_Degree_relative", "Hamilton_Scale",
    "Standard_Alcoholunits_Last_28days",
    "TAP_A_1", "TAP_A_2", "TAP_A_3", "TAP_A_4", "TAP_A_5", "TAP_A_6", "TAP_A_7", "TAP_A_8", "TAP_A_9", "TAP_A_10", "TAP_A_11", "TAP_A_12", "TAP_A_13", "TAP_A_14",
    "TAP_A_15", "TAP_A_16"
]
data=data[relevant_columns]

numeric_columns=[
    "Hamilton_Scale", "Standard_Alcoholunits_Last_28days",
    "TAP_A_1", "TAP_A_2", "TAP_A_3", "TAP_A_4",
    "TAP_A_5", "TAP_A_6", "TAP_A_7", "TAP_A_8", "TAP_A_9",
    "TAP_A_10", "TAP_A_11", "TAP_A_12", "TAP_A_13", "TAP_A_14",
    "TAP_A_15", "TAP_A_16"
]

for col in numeric_columns:
    data[col]=data[col].astype(str).str.replace(',', '.')
    data[col]=pd.to_numeric(data[col], errors='coerce')

imputer=SimpleImputer(strategy='mean')
data[numeric_columns]=imputer.fit_transform(data[numeric_columns])

scaler=StandardScaler()
data[numeric_columns]=scaler.fit_transform(data[numeric_columns])

plt.figure(figsize=(16, 12))

for i, col in enumerate(numeric_columns, 1):
    plt.subplot(6, 4, i)
    sns.histplot(data[col], kde=True)
    plt.title(f"Original - {col}")

plt.tight_layout()
plt.show()

scaler=MinMaxScaler()
data[numeric_columns]=scaler.fit_transform(data[numeric_columns])

plt.figure(figsize=(16, 12))

for i, col in enumerate(numeric_columns, 1):
    plt.subplot(6, 4, i)
    sns.histplot(data[col], kde=True)
    plt.title(f"Normalized - {col}")
plt.tight_layout()
plt.show()

imputer=SimpleImputer(strategy='mean')
data[numeric_columns]=imputer.fit_transform(data[numeric_columns])

category_columns=[
    "Gender", "Education", "DRUG", "DRUG_0=negative_1=Positive", "Drug_Substance", "Smoking", "Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)",
    "SKID_Diagnoses", "SKID_Diagnoses 1", "SKID_Diagnoses 2", "Comments_SKID_assessment", "Alcohol_Dependence_In_1st-3rd_Degree_relative"
]
encoder=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
data_encoded=pd.DataFrame(encoder.fit_transform(data[category_columns]))
data_encoded.columns=encoder.get_feature_names_out(category_columns)

data=pd.concat([data.drop(columns=category_columns), data_encoded], axis=1)

print("Original Data Stats:\n", data[numeric_columns].describe())

scaler=StandardScaler()
normalized_data=pd.DataFrame(scaler.fit_transform(data[numeric_columns]), columns=numeric_columns)

print("\nNormalized Data Stats:\n", normalized_data.describe())
print(data.describe())

plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(6, 4, i)
    sns.histplot(data[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

numeric_data=data.drop(columns=['ID'], errors='ignore')
correlation_matrix=numeric_data.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

pca=PCA(n_components=2)
principal_components=pca.fit_transform(data[numeric_columns])
data['PCA1']=principal_components[:, 0]
data['PCA2']=principal_components[:, 1]
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='TAP_A_15', data=data)
plt.title('PCA Scatter Plot')
plt.show()

target=["TAP_A_1", "TAP_A_2", "TAP_A_3", "TAP_A_4", "TAP_A_5", "TAP_A_6", "TAP_A_7", "TAP_A_8", "TAP_A_9", "TAP_A_10", "TAP_A_11", "TAP_A_12", "TAP_A_13", "TAP_A_14", "TAP_A_15", "TAP_A_16"]
features=[col for col in data.columns if col not in [target, 'ID']]

X=data[features]
y=data[target]

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

model=RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print(f"mean squared error : {mse:.2f}")
print(f"r2 value : {r2:.2f}")

feature_importances=pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
feature_importances=feature_importances.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance')
plt.show()