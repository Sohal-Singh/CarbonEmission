from preprocessing import df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

print("\nFinal Dataset:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

x = df.drop(columns=['CarbonEmission', 'Emission Category'])
y = df['Emission Category']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print("\nTraining Set Shape:")
print(X_train.shape, y_train.shape)

model = RandomForestClassifier(n_estimators=300, random_state=0, min_samples_leaf=2, n_jobs=-1)
model.fit(X_train, y_train)
joblib.dump(model, "Models/random_forest_model.pkl")

y_pred = model.predict(X_test)

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))