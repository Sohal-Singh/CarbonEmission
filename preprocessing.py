import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

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

'''
ENCODER FIXES REQUIRED:
1) The LabelEncoder's states are not being saved, only the last column's state is saved. 
To fix this, we need to save the encoder's state for each categorical column separately.

2) The columns "Recycling" and "Cooking_With" are not being encoded correctly.
To fix: Use MultiLabelBinarizer for these columns instead of LabelEncoder, as they contain multiple labels per entry.
'''

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column].astype(str))

joblib.dump(encoder, "Models/label_encoder.pkl")

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
print(encoder.classes_)
