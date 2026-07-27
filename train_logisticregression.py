from preprocessing import df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("\nFinal Dataset:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

x = df.drop(columns=['CarbonEmission', 'Emission Category'])
y = df["Emission Category"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nTraining Set Shape:")
print(X_train.shape, y_train.shape)

model = LogisticRegression(max_iter=1000, n_jobs=-1)
model.fit(X_train, y_train)
joblib.dump(model, "Models/logistic_regression_model.pkl")

y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))