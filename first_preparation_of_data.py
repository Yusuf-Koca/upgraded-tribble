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

# If you need me to be honest this file is A MESS. Issue is, all this code was written in Jupiter Notebook (as you can probably tell by the line of commented code
# above) because I like it better, but now since some time has passed I don't really remember what I did and did not end up using. This is TRAGIC and I would prefer
# to not even show it, but I spoke about the process during the ML presentation so I think I should also show it. BUT, because I thought it's so bad, that is why 
# when we needed to write the code together, as a group, I decided to simplify it and make it more elegant than THIS. I did my best trying to put comments here but 
# truly, this code is a product of my desperation.

# Of course first I needed to load the data, I used an odf file, but for the final one I converted to csv to make it look more "professional".
file_path='./manual.ods'
data=pd.read_excel(file_path, engine='odf')

# HERE THE PREPROCESSING BEGINS

# As you might remember, first we had this big idea for our project and included all these metrics, but I started to get lost in the data and stopped understanding
# what is what and how to interpret it.
relevant_columns=[
    "ID", "Gender", "Education", "DRUG", "DRUG_0=negative_1=Positive", "Drug_Substance","Smoking", "Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)",
    "SKID_Diagnoses", "SKID_Diagnoses 1", "SKID_Diagnoses 2", "Comments_SKID_assessment", "Alcohol_Dependence_In_1st-3rd_Degree_relative", "Hamilton_Scale",
    "Standard_Alcoholunits_Last_28days",
    "TAP_A_1", "TAP_A_2", "TAP_A_3", "TAP_A_4", "TAP_A_5", "TAP_A_6", "TAP_A_7", "TAP_A_8", "TAP_A_9", "TAP_A_10", "TAP_A_11", "TAP_A_12", "TAP_A_13", "TAP_A_14",
    "TAP_A_15", "TAP_A_16"
]
# Subsetting the data.
data=data[relevant_columns]

# As we needed to convert some data to numeric values, because they were saved as strings, here's that. In the loop I also replaced comas with periods so it can
# be done. Otherwise it was an issue.
numeric_columns=[
    "Hamilton_Scale", "Standard_Alcoholunits_Last_28days",
    "TAP_A_1", "TAP_A_2", "TAP_A_3", "TAP_A_4",
    "TAP_A_5", "TAP_A_6", "TAP_A_7", "TAP_A_8", "TAP_A_9",
    "TAP_A_10", "TAP_A_11", "TAP_A_12", "TAP_A_13", "TAP_A_14",
    "TAP_A_15", "TAP_A_16"
]

for col in numeric_columns:
    data[col]=data[col].astype(str).str.replace(',', '.')
# Line below I ended up using in the final code, it replaces non-numeric (as in words) values with NaN.
    data[col]=pd.to_numeric(data[col], errors='coerce')

# Code below is handling missing data
imputer=SimpleImputer(strategy='mean')
data[numeric_columns]=imputer.fit_transform(data[numeric_columns])

# Normalising these numeric columns. Here I think I started to have so trouble and didn't understand what was going on, because I was getting the same graphs
# before and after normalisation.
scaler=StandardScaler()
data[numeric_columns]=scaler.fit_transform(data[numeric_columns])

# So I first plotted the original data, before I normalised it.
plt.figure(figsize=(16, 12))

for i, col in enumerate(numeric_columns, 1):
    plt.subplot(6, 4, i)
    sns.histplot(data[col], kde=True)
    plt.title(f"Original - {col}")

plt.tight_layout()
plt.show()

# Then I re-run the code from lines 60 and 61, but I was still unsatisfied so I just decided to use a different stadarisation as my normalisation.
scaler=MinMaxScaler()
data[numeric_columns]=scaler.fit_transform(data[numeric_columns])

# And then plotted the newly normalised data but I am no longer sure of what the outcome was.
plt.figure(figsize=(16, 12))

for i, col in enumerate(numeric_columns, 1):
    plt.subplot(6, 4, i)
    sns.histplot(data[col], kde=True)
    plt.title(f"Normalized - {col}")
plt.tight_layout()
plt.show()

# Doing same thing again with missing data.
imputer=SimpleImputer(strategy='mean')
data[numeric_columns]=imputer.fit_transform(data[numeric_columns])

# Code below is handling the encoding of cathegorical columns- I used OneHotEncoder- something new for me.
category_columns=[
    "Gender", "Education", "DRUG", "DRUG_0=negative_1=Positive", "Drug_Substance", "Smoking", "Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)",
    "SKID_Diagnoses", "SKID_Diagnoses 1", "SKID_Diagnoses 2", "Comments_SKID_assessment", "Alcohol_Dependence_In_1st-3rd_Degree_relative"
]
encoder=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
data_encoded=pd.DataFrame(encoder.fit_transform(data[category_columns]))
data_encoded.columns=encoder.get_feature_names_out(category_columns)

# And so at last I had to combine both the numerical and my encoded cathegorical data.
data=pd.concat([data.drop(columns=category_columns), data_encoded], axis=1)

print("Original Data Stats:\n", data[numeric_columns].describe())

# Once again normalisation...
scaler=StandardScaler()
normalized_data=pd.DataFrame(scaler.fit_transform(data[numeric_columns]), columns=numeric_columns)

print("\nNormalized Data Stats:\n", normalized_data.describe())

# HERE THE DATA EXPLORATION BEGINS

# Of course first thing first, I printed the descriptive statistics.
print(data.describe())

# Plotting the distribution plots.
plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(6, 4, i)
    sns.histplot(data[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Had to exclude the non-numeric columns from correlation analysis.
numeric_data=data.drop(columns=['ID'], errors='ignore')
correlation_matrix=numeric_data.corr()

# And then plot the correletion matrix that was as big as the country of France and from which we decided to exclude a lot of data (even the ones that would 
# have been usefull, which we knew but decided to do so anyway because it was hard- simply, we decided we first need to learn before jumping into so much data).
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Below is the dimensionality reduction using PCA, which we ended up not using in our final project because it was so scatered.
pca=PCA(n_components=2)
principal_components=pca.fit_transform(data[numeric_columns])
data['PCA1']=principal_components[:, 0]
data['PCA2']=principal_components[:, 1]
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='TAP_A_15', data=data)
plt.title('PCA Scatter Plot')
plt.show()

# HERE MY LITTLE INVESTIGATION INTO RELATIONSHIPS BEGINS

# First, I defined the targets and features.
target=["TAP_A_1", "TAP_A_2", "TAP_A_3", "TAP_A_4", "TAP_A_5", "TAP_A_6", "TAP_A_7", "TAP_A_8", "TAP_A_9", "TAP_A_10", "TAP_A_11", "TAP_A_12", "TAP_A_13", "TAP_A_14", "TAP_A_15", "TAP_A_16"]
features=[col for col in data.columns if col not in [target, 'ID']]

X=data[features]
y=data[target]

# Trained-test splited. I know that setting random state to 42 is an old meme but I like it so this random state is also 42.
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

# Then trained a RandomForest model.
model=RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# This line of code is for predictions.
y_pred=model.predict(X_test)

# Here I was evaluating the model.
mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print(f"mean squared error : {mse:.2f}")
print(f"r2 value : {r2:.2f}")

# And this is feature importance that I basically plotted for nothing.
feature_importances=pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
feature_importances=feature_importances.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance')
plt.show()

# I'm really sorry if you read though it. I know the code in this file is abysmal.