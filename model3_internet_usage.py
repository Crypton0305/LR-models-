import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

data = pd.read_csv("data_for_ml.csv")

data = data.dropna(subset=['Final_Score'])
data['Internet_Usage'] = data['Internet_Usage'].fillna(data['Internet_Usage'].median())
data = data.dropna(subset=['Internet_Usage'])

X = data[["Internet_Usage"]]
y = data["Final_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler3 = StandardScaler()
X_train_scale = scaler3.fit_transform(X_train)
X_test_scale  = scaler3.transform(X_test)

model3 = LinearRegression()
model3.fit(X_train_scale, y_train)

y_pred = model3.predict(X_test_scale)
r2 = r2_score(y_test, y_pred)
print("Model 3  |  Internet Usage → Final Score")
print("R2 Score:", r2)

joblib.dump(model3,  "model3_internet")
joblib.dump(scaler3, "scaler3_internet")
print("Model saved!")

def predict_score_by_internet():
    internet = float(input("Enter Internet Usage (hours): "))
    scaled = scaler3.transform([[internet]])
    result = model3.predict(scaled)
    print("Predicted Final Score:", result[0])

predict_score_by_internet()

loaded_model3  = joblib.load("model3_internet")
loaded_scaler3 = joblib.load("scaler3_internet")
print("Model 3 loaded!")

def predict_by_internet():
    internet = float(input("Enter Internet Usage (hours): "))
    scaled = loaded_scaler3.transform([[internet]])
    result = loaded_model3.predict(scaled)
    print("Predicted Final Score:", result[0])

predict_by_internet()
