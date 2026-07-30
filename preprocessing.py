import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import ast
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
df["Cooking_With"] = df["Cooking_With"].apply(ast.literal_eval)
df["Recycling"] = df["Recycling"].apply(ast.literal_eval)

print("\nMissing Values After Filling:")
print(df.isnull().sum())

print("\nCategorical Columns:")
print(df.select_dtypes(include="object").columns)
cooking_encoder = MultiLabelBinarizer()

cooking_encoded = cooking_encoder.fit_transform(df["Cooking_With"])

cooking_df = pd.DataFrame(
    cooking_encoded,
    columns=["Cooking_" + x for x in cooking_encoder.classes_],
    index=df.index
)


recycling_encoder = MultiLabelBinarizer()

recycling_encoded = recycling_encoder.fit_transform(df["Recycling"])

recycling_df = pd.DataFrame(
    recycling_encoded,
    columns=["Recycle_" + x for x in recycling_encoder.classes_],
    index=df.index
)
df = pd.concat(
    [df, cooking_df, recycling_df],
    axis=1
)
df.drop(
    ["Cooking_With", "Recycling"],
    axis=1,
    inplace=True
)
categorical_columns = df.select_dtypes(include="object").columns

onehot_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_data = onehot_encoder.fit_transform(df[categorical_columns])

encoded_df = pd.DataFrame(
    encoded_data,
    columns=onehot_encoder.get_feature_names_out(categorical_columns),
    index=df.index
)

df = pd.concat(
    [df.drop(columns=categorical_columns), encoded_df],
    axis=1
)

joblib.dump(cooking_encoder,"Models/cooking_encoder.joblib")
joblib.dump(recycling_encoder,"Models/recycling_encoder.joblib")
joblib.dump(onehot_encoder,"Models/onehot_encoder.joblib")

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
