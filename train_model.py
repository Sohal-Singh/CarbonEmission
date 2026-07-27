from preprocessing import df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("\nFinal Dataset:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

x = df.drop(columns=['CarbonEmission', 'Emission Category'])
y = df['CarbonEmission']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nTraining Set Shape:")
print(X_train.shape, y_train.shape)

model = LinearRegression(n_jobs=-1)
model.fit(X_train, y_train)
joblib.dump(model, "Models/linear_regression_model.pkl")

y_pred = model.predict(X_test)

print("\nMean Absolute Error:")
print(mean_absolute_error(y_test, y_pred))

print("\nMean Squared Error:")
print(mean_squared_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))