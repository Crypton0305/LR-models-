import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

data = pd.read_csv("data_for_ml.csv")

data = data.dropna(subset=['Final_Score'])
data['Attendance'] = data['Attendance'].fillna(data['Attendance'].median())
data = data.dropna(subset=['Attendance'])

X = data[["Attendance"]]
y = data["Final_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler4 = StandardScaler()
X_train_scale = scaler4.fit_transform(X_train)
X_test_scale  = scaler4.transform(X_test)

model4 = LinearRegression()
model4.fit(X_train_scale, y_train)

y_pred = model4.predict(X_test_scale)
r2 = r2_score(y_test, y_pred)
print("Model 4  |  Attendance → Final Score")
print("R2 Score:", r2)

joblib.dump(model4,  "model4_attendance")
joblib.dump(scaler4, "scaler4_attendance")
print("Model saved!")

def predict_score_by_attendance():
    attendance = float(input("Enter Attendance (%): "))
    scaled = scaler4.transform([[attendance]])
    result = model4.predict(scaled)
    print("Predicted Final Score:", result[0])

predict_score_by_attendance()

loaded_model4  = joblib.load("model4_attendance")
loaded_scaler4 = joblib.load("scaler4_attendance")
print("Model 4 loaded!")

def predict_by_attendance():
    attendance = float(input("Enter Attendance (%): "))
    scaled = loaded_scaler4.transform([[attendance]])
    result = loaded_model4.predict(scaled)
    print("Predicted Final Score:", result[0])

predict_by_attendance()
