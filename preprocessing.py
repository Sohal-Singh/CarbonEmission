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

bins = [0, 1538, 2768, float("inf")]
labels = ["Low", "Medium", "High"]

df["CarbonEmission"] = pd.cut(
    df["CarbonEmission"],
    bins=bins,
    labels=labels
)
print(df.head())