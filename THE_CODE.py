import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# The function below is responsible for preprocessing, as the name suggests.
def preprocess(file_path):
# First thing first, we have to load the data from the file.
    data=pd.read_csv(file_path)

# Here, we're selecting columns we need for the analysis: smoking and drug usage and all the attention metrics from TAP Alertness Questionnaire
    columns_of_interest=['Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)', 'DRUG_0=negative_1=Positive']+[f'TAP_A_{i}' for i in range(1, 17)]
    data_subset=data[columns_of_interest].copy()

# In order to obtain better clarity of our data, this function renames the columns into shorter, more readable names.
    data_subset.rename(columns={
        'Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)': 'Smoking', 'DRUG_0=negative_1=Positive': 'Drug_Use'}, inplace=True)

# The line below is dropping all the rows with NaN values, so we don't analyse "ghosts".
    data_subset.dropna(inplace=True)

# Three lines below are doing virtually the same things (error handling -> turning them to NaN values and dropping these invalid rows),
# but firstly it converts our attention metrics to numeric because they were saved as strings.
    attention_metrics=[f'TAP_A_{i}' for i in range(1, 17)]
    data_subset[attention_metrics]=data_subset[attention_metrics].apply(pd.to_numeric, errors='coerce')
    data_subset.dropna(inplace=True)

# Code under here is standarising the data. Don't really know what to say more about that.
    scaler=StandardScaler()
    data_subset_scaled=scaler.fit_transform(data_subset)
    data_subset_scaled=pd.DataFrame(data_subset_scaled, columns=data_subset.columns)

    return data_subset_scaled, attention_metrics

# Our second function in this code handles data exploration.
def exploring(data):
# Below we decided to print out descriptive statistics. Just like we did in class.
    print("descriptive stats: ")
    print(data.describe())

#Here is a plot for feature distributions.
    plt.figure(figsize=(12, 6))
    data.hist(column=['Smoking', 'Drug_Use'], bins=10, layout=(1, 2), figsize=(12, 4))
    plt.suptitle("Feature Distributions")
    plt.show()

# Here is a correletion heatmap.
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

#function below performs clustering with hyperparametere tuning
def perform_clustering(data, attention_metrics):
    features = ['Smoking', 'Drug_Use'] + attention_metrics #firstly we have to define features for clustering
    X = data[features]
#we determine the optimal number of clusters with elbow method
    wcss = []
    cluster_range = range(2, 21) #went with this number of clusters
#finetuning
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1) #42 for the meaning of life of course
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
#below we are ploting a graph for elbow method
    plt.figure(figsize=(8, 6))
    plt.plot(cluster_range, wcss, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.grid()
    plt.show()
#below hyperparameter tuning using grid search
    param_grid = {'n_clusters': cluster_range, 'n_init': [1, 2, 3]}
    grid_search = GridSearchCV(KMeans(random_state=42), param_grid, cv=5)
    grid_search.fit(X)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
#and then training our Kmeans model with optimal number of clusters
    optimal_clusters = best_params['n_clusters']
    final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=best_params['n_init'])
    data['Cluster'] = final_kmeans.fit_predict(X)

    return data, optimal_clusters

#function below analyzes and visualizes our clusters
def analyze_clusters(data, attention_metrics):
#used a scatter plot for these selected features
    plt.figure(figsize=(12, 10))
    sns.pairplot(
        data,
        vars=['Smoking', 'Drug_Use'] + attention_metrics[:2], 
        hue='Cluster',
        palette='viridis',
        diag_kind='kde'
    )
    plt.suptitle('Cluster Visualization with Selected Features', y=1.02)
    plt.show()
#summorizing clusters
    cluster_summary = data.groupby('Cluster')[['Smoking', 'Drug_Use'] + attention_metrics].mean()
    print("Cluster Summary:")
    print(cluster_summary)

#this is the main function to execute everything one by one
def main():
    file_path = 'manual.csv'
    data, attention_metrics = preprocess(file_path) #preprocessing data
    exploring(data) #exploring data
    data, optimal_clusters = perform_clustering(data, attention_metrics) #clustering
    analyze_clusters(data, attention_metrics) #actual analysis and visualization

#running everything
if __name__ == "__main__":
    main()