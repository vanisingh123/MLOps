import pandas as pd

def load_data(path):
    col_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    df = pd.read_csv(path, header=None, names=col_names, na_values=' ?', skipinitialspace=True)
    df.dropna(inplace=True)
    return df

def preprocess_data(df):
    df = pd.get_dummies(df, drop_first=True)
    return df
