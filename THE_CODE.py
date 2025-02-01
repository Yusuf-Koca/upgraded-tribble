import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

def preprocess(file_path):
    data=pd.read_csv(file_path)

    columns_of_interest=['Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)', 'DRUG_0=negative_1=Positive']+[f'TAP_A_{i}' for i in range(1, 17)]
    data_subset=data[columns_of_interest].copy()

    data_subset.rename(columns={
        'Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)': 'Smoking', 'DRUG_0=negative_1=Positive': 'Drug_Use'}, inplace=True)
    data_subset.dropna(inplace=True)

    attention_metrics=[f'TAP_A_{i}' for i in range(1, 17)]
    data_subset[attention_metrics]=data_subset[attention_metrics].apply(pd.to_numeric, errors='coerce')
    data_subset.dropna(inplace=True)

    scaler=StandardScaler()
    data_subset_scaled=scaler.fit_transform(data_subset)
    data_subset_scaled=pd.DataFrame(data_subset_scaled, columns=data_subset.columns)

    return data_subset_scaled, attention_metrics

def exploring(data):
    print("descriptive stats: ")
    print(data.describe())

    plt.figure(figsize=(12, 6))
    data.hist(column=['Smoking', 'Drug_Use'], bins=10, layout=(1, 2), figsize=(12, 4))
    plt.suptitle("Feature Distributions")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def perform_clustering(data, attention_metrics):
    features = ['Smoking', 'Drug_Use'] + attention_metrics
    X = data[features]
    wcss = []
    cluster_range = range(2, 21)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1) 
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(cluster_range, wcss, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.grid()
    plt.show()

    param_grid = {'n_clusters': cluster_range, 'n_init': [1, 2, 3]}
    grid_search = GridSearchCV(KMeans(random_state=42), param_grid, cv=5)
    grid_search.fit(X)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    optimal_clusters = best_params['n_clusters']
    final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=best_params['n_init'])
    data['Cluster'] = final_kmeans.fit_predict(X)

    return data, optimal_clusters

def analyze_clusters(data, attention_metrics):
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
    cluster_summary = data.groupby('Cluster')[['Smoking', 'Drug_Use'] + attention_metrics].mean()
    print("Cluster Summary:")
    print(cluster_summary)

def main():
    file_path = 'manual.csv'
    data, attention_metrics = preprocess(file_path)
    exploring(data)
    data, optimal_clusters = perform_clustering(data, attention_metrics)
    analyze_clusters(data, attention_metrics)

if __name__ == "__main__":
    main()