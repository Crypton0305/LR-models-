import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

data = pd.read_csv("data_for_ml.csv")

data = data.dropna(subset=['Final_Score'])
data['Sleep_Hours'] = data['Sleep_Hours'].fillna(data['Sleep_Hours'].median())
data = data.dropna(subset=['Sleep_Hours'])

X = data[["Sleep_Hours"]]
y = data["Final_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler2 = StandardScaler()
X_train_scale = scaler2.fit_transform(X_train)
X_test_scale  = scaler2.transform(X_test)

model2 = LinearRegression()
model2.fit(X_train_scale, y_train)

y_pred = model2.predict(X_test_scale)
r2 = r2_score(y_test, y_pred)
print("Model 2  |  Sleep Hours → Final Score")
print("R2 Score:", r2)

joblib.dump(model2,  "model2_sleep")
joblib.dump(scaler2, "scaler2_sleep")
print("Model saved!")

def predict_score_by_sleep():
    sleep = float(input("Enter Sleep Hours: "))
    scaled = scaler2.transform([[sleep]])
    result = model2.predict(scaled)
    print("Predicted Final Score:", result[0])

predict_score_by_sleep()

loaded_model2  = joblib.load("model2_sleep")
loaded_scaler2 = joblib.load("scaler2_sleep")
print("Model 2 loaded!")

def predict_by_sleep():
    sleep = float(input("Enter Sleep Hours: "))
    scaled = loaded_scaler2.transform([[sleep]])
    result = loaded_model2.predict(scaled)
    print("Predicted Final Score:", result[0])

predict_by_sleep()
