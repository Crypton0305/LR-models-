# 🎓 Student Final Score Prediction — Linear Regression Models

This project contains 5 individual **Simple Linear Regression** models that predict a student's **Final Score** — each model uses only **1 input feature**.

---

## 📁 Project Structure

```
├── data_for_ml.csv               # Dataset
├── model1_hours_studied.py       # Model 1 — Hours Studied
├── model2_sleep_hours.py         # Model 2 — Sleep Hours
├── model3_internet_usage.py      # Model 3 — Internet Usage
├── model4_attendance.py          # Model 4 — Attendance
├── model5_gender.py              # Model 5 — Gender
├── requirements.txt              # Required libraries
└── README.md
```

---

## 📊 Dataset

| Column | Description |
|--------|-------------|
| `Hours_Studied` | Daily study hours |
| `Sleep_Hours` | Daily sleep hours |
| `Internet_Usage` | Daily internet usage hours |
| `Gender` | Male / Female |
| `Study_Method` | Self Study / Coaching / Group Study |
| `Attendance` | School attendance percentage |
| `Final_Score` | **Target variable** — final exam score |

---

## 🤖 Models

| Model | Input Feature | Output |
|-------|--------------|--------|
| Model 1 | Hours Studied | Final Score |
| Model 2 | Sleep Hours | Final Score |
| Model 3 | Internet Usage | Final Score |
| Model 4 | Attendance | Final Score |
| Model 5 | Gender (Encoded) | Final Score |

---

## ⚙️ How Each Model Works

Every `.py` file follows the same steps:

1. **Load Data** — Read CSV file
2. **Cleaning** — Fill or drop NaN values
3. **X and y** — Separate input and output
4. **Train Test Split** — 70% train, 30% test
5. **StandardScaler** — Scale the data
6. **LinearRegression** — Train the model
7. **R2 Score** — Evaluate the model
8. **joblib.dump** — Save the model
9. **joblib.load** — Reload the saved model
10. **Predict Function** — Take input and predict score

---

## 🚀 How to Run

### Install requirements:
```bash
pip install -r requirements.txt
```

### Run any model:
```bash
python model1_hours_studied.py
python model2_sleep_hours.py
python model3_internet_usage.py
python model4_attendance.py
python model5_gender.py
```

### Example:
```
Enter Hours Studied: 7
Predicted Final Score: 76.43
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
joblib
```

---

## 👨‍💻 Tech Stack

- **Python 3.x**
- **Pandas** — Data loading & cleaning
- **Scikit-learn** — ML models, scaling, evaluation
- **Joblib** — Model save & load

---

## 📝 Notes

- Dataset had some missing values and typos (`Femle`, `male`) which are handled automatically during cleaning
- `Gender` column is encoded using `LabelEncoder` (Model 5)
- All models use `StandardScaler` for feature scaling
- Saved model files are stored in the same folder