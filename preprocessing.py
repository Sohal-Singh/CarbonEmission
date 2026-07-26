import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("CarbonEmission.csv")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)
print("\nShape After Removing Duplicates:")
print(df.shape)

df["Vehicle Type"] = df["Vehicle Type"].fillna("None")
print("\nMissing Values After Filling:")
print(df.isnull().sum())

print("\nCategorical Columns:")
print(df.select_dtypes(include="object").columns)

encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column].astype(str))

print("\nEncoded Dataset:")
print(df.head())
print(df["CarbonEmission"].describe())

low_limit = df["CarbonEmission"].quantile(0.33)
high_limit = df["CarbonEmission"].quantile(0.66)

def emission_category(value):
    if value <= low_limit:
        return 0
    elif value <= high_limit:
        return 1
    else:
        return 2

df["Emission Category"] = df["CarbonEmission"].apply(emission_category)
print(df[["CarbonEmission", "Emission Category"]].head())
print(df["Emission Category"].value_counts())


