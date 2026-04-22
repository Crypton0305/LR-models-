# ============================================================
# Model 5 — Gender → Final Score
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# ── Load Data ─────────────────────────────────────────────
data = pd.read_csv("data_for_ml.csv")

# ── Cleaning ──────────────────────────────────────────────
data = data.dropna(subset=['Final_Score'])
data['Gender'] = data['Gender'].str.strip().str.capitalize().replace({'Femle': 'Female'})
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])

# ── Encode Gender ─────────────────────────────────────────
data['Gender_enc'] = LabelEncoder().fit_transform(data['Gender'])

# ── X and y ───────────────────────────────────────────────
X = data[["Gender_enc"]]
y = data["Final_Score"]

# ── Train Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ── Scaling ───────────────────────────────────────────────
scaler5 = StandardScaler()
X_train_scale = scaler5.fit_transform(X_train)
X_test_scale  = scaler5.transform(X_test)

# ── Model ─────────────────────────────────────────────────
model5 = LinearRegression()
model5.fit(X_train_scale, y_train)

# ── Evaluation ────────────────────────────────────────────
y_pred = model5.predict(X_test_scale)
r2 = r2_score(y_test, y_pred)
print("Model 5  |  Gender → Final Score")
print("R2 Score:", r2)

# ── Save ──────────────────────────────────────────────────
joblib.dump(model5,  "model5_gender")
joblib.dump(scaler5, "scaler5_gender")
print("Model saved!")

# ── Predict Function ──────────────────────────────────────
def predict_score_by_gender():
    gender = input("Enter Gender (Male/Female): ").strip().capitalize()
    enc = 1 if gender == "Male" else 0
    scaled = scaler5.transform([[enc]])
    result = model5.predict(scaled)
    print("Predicted Final Score:", result[0])

predict_score_by_gender()

# ── Load & Predict ────────────────────────────────────────
loaded_model5  = joblib.load("model5_gender")
loaded_scaler5 = joblib.load("scaler5_gender")
print("Model 5 loaded!")

def predict_by_gender():
    gender = input("Enter Gender (Male/Female): ").strip().capitalize()
    enc = 1 if gender == "Male" else 0
    scaled = loaded_scaler5.transform([[enc]])
    result = loaded_model5.predict(scaled)
    print("Predicted Final Score:", result[0])

predict_by_gender()
