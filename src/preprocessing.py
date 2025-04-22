import pandas as pd

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Load raw data
df = pd.read_csv("data/adult.data", names=columns, na_values=" ?", skipinitialspace=True)

# Basic preprocessing: drop missing values
df.dropna(inplace=True)

# Save cleaned data
df.to_csv("data/adult_cleaned.csv", index=False)
df.to_csv("data/processed.csv", index=False)