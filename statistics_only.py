import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

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

    data_subset['TAP_Attention']=data_subset[attention_metrics].mean(axis=1)

    scaler=StandardScaler()
    data_subset_scaled=scaler.fit_transform(data_subset)
    data_subset_scaled=pd.DataFrame(data_subset_scaled, columns=data_subset.columns)

    return data_subset_scaled, data_subset

def desc_stat(data):
    print("descriptive stats: ")
    print(data.describe())

def hypothesis(data):
    smoking_groups=data.groupby('Smoking')['TAP_Attention']
    non_smoker=smoking_groups.get_group(1)
    occasional_smoker=smoking_groups.get_group(2)
    smoker=smoking_groups.get_group(3)
    
    f_val, p_val=stats.f_oneway(non_smoker, occasional_smoker, smoker)
    print(f"anova F-value: {f_val:.3f}, p-value: {p_val:.3f}")

def regression(data):
    X = data[['Smoking', 'Drug_Use']]
    y = data['TAP_Attention']
    model = LinearRegression()
    model.fit(X, y)
    print(f"regression coeficientes: {model.coef_}")
    print(f"r2 value: {model.score(X, y):.3f}")

def intervals(data):
    mean_tap=data['TAP_Attention'].mean()
    std_tap=data['TAP_Attention'].std()
    n=len(data['TAP_Attention'])
    c_interval=stats.t.interval(0.95, df=n-1, loc=mean_tap, scale=std_tap/np.sqrt(n))
    print(f"95% confidence interval: {c_interval}")

def whole_ass_analysis(file_path):
    data_scaled, original_data = preprocess(file_path)
    desc_stat(original_data)
    hypothesis(original_data)
    regression(original_data)
    intervals(original_data)

if __name__ == "__main__":
    file_path = 'manual.csv'
    whole_ass_analysis(file_path)