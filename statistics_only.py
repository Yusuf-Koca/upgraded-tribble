import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# Here I decided to just pretty much copy the preprocessing from the main file, as unfortunatelly I'm very particular and wanted this file 
# to function as a stand-alone file- just in case. So for more information on each line, I invite you to explore the main file called "THE_CODE",
# because here I really just changed on line that I commented under.
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

# The line under this comment is the only new one.
# Its purpose is to calculate the mean of all columns from TAP-Alertness Questionnaire that we summorised as "Attention", this is purely so we
# don't analise every single metric of the questionnaire alone but see it as attention, as a whole.
    data_subset['TAP_Attention']=data_subset[attention_metrics].mean(axis=1)

    scaler=StandardScaler()
    data_subset_scaled=scaler.fit_transform(data_subset)
    data_subset_scaled=pd.DataFrame(data_subset_scaled, columns=data_subset.columns)

    return data_subset_scaled, data_subset

# This part is also 1:1 from the main file ("THE_CODE"), part of exploring that we did.
def desc_stat(data):
    print("descriptive stats: ")
    print(data.describe())

# This funcion tests our hypothesis. I don't know what to say about it tobe honest, it's just statistics I learned in other projects.
def hypothesis(data):
    smoking_groups=data.groupby('Smoking')['TAP_Attention']
    non_smoker=smoking_groups.get_group(1)
    occasional_smoker=smoking_groups.get_group(2)
    smoker=smoking_groups.get_group(3)

# We performed ANOVA test. Again, no idea what to say about that, it's just a Python implementation of this particular statistical test.
    f_val, p_val=stats.f_oneway(non_smoker, occasional_smoker, smoker)
    print(f"anova F-value: {f_val:.3f}, p-value: {p_val:.3f}")

# This function is performing regression, just like we did in class. I wanted to have it for these "scientific" purposes in our ML presentation.
# Here, from what I remember, I just copied the code from our classes (adjusted for this analysis of course).
def regression(data):
    X = data[['Smoking', 'Drug_Use']]
    y = data['TAP_Attention']
    model = LinearRegression()
    model.fit(X, y)
    print(f"regression coeficientes: {model.coef_}")
    print(f"r2 value: {model.score(X, y):.3f}")

# Additionally I decided to check out the confidence intervals, to get the estimate on the range of values for that fall, but that we talked about
# at the ML presentation. Again, it's just statistics.
def intervals(data):
    mean_tap=data['TAP_Attention'].mean()
    std_tap=data['TAP_Attention'].std()
    n=len(data['TAP_Attention'])
    c_interval=stats.t.interval(0.95, df=n-1, loc=mean_tap, scale=std_tap/np.sqrt(n))
    print(f"95% confidence interval: {c_interval}")

# First of all I'm very sorry about the previous name of this function below- I hope you did not see it. This is pretty much the same as the main file, so it can 
# function as stand-alone file and execute itself properly.
def analysis(file_path):
    data_scaled, original_data = preprocess(file_path)
    desc_stat(original_data)
    hypothesis(original_data)
    regression(original_data)
    intervals(original_data)

# Here we're running the analysis and telling other functions where to take the data from, as in from which file.
if __name__ == "__main__":
    file_path = 'manual.csv'
    analysis(file_path)