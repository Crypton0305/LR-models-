# ============================================================
# Model 1 — Hours Studied → Final Score
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# ── Load Data ─────────────────────────────────────────────
data = pd.read_csv("data_for_ml.csv")

# ── Cleaning ──────────────────────────────────────────────
data = data.dropna(subset=['Final_Score'])
data['Hours_Studied'] = data['Hours_Studied'].fillna(data['Hours_Studied'].median())
data = data.dropna(subset=['Hours_Studied'])

# ── X and y ───────────────────────────────────────────────
X = data[["Hours_Studied"]]
y = data["Final_Score"]

# ── Train Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ── Scaling ───────────────────────────────────────────────
scaler1 = StandardScaler()
X_train_scale = scaler1.fit_transform(X_train)
X_test_scale  = scaler1.transform(X_test)

# ── Model ─────────────────────────────────────────────────
model1 = LinearRegression()
model1.fit(X_train_scale, y_train)

# ── Evaluation ────────────────────────────────────────────
y_pred = model1.predict(X_test_scale)
r2 = r2_score(y_test, y_pred)
print("Model 1  |  Hours Studied → Final Score")
print("R2 Score:", r2)

# ── Save ──────────────────────────────────────────────────
joblib.dump(model1,  "model1_hours")
joblib.dump(scaler1, "scaler1_hours")
print("Model saved!")

# ── Predict Function ──────────────────────────────────────
def predict_score_by_hours():
    hours = float(input("Enter Hours Studied: "))
    scaled = scaler1.transform([[hours]])
    result = model1.predict(scaled)
    print("Predicted Final Score:", result[0])

predict_score_by_hours()

# ── Load & Predict ────────────────────────────────────────
loaded_model1  = joblib.load("model1_hours")
loaded_scaler1 = joblib.load("scaler1_hours")
print("Model 1 loaded!")

def predict_by_hours():
    hours = float(input("Enter Hours Studied: "))
    scaled = loaded_scaler1.transform([[hours]])
    result = loaded_model1.predict(scaled)
    print("Predicted Final Score:", result[0])

predict_by_hours()
