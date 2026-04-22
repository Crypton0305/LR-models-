# 🎓 Student Final Score Prediction — Linear Regression Models

Yeh project 5 alag alag **Simple Linear Regression** models contain karta hai jo ek student ka **Final Score** predict karte hain — har model mein sirf **1 input feature** use hoti hai.

---

## 📁 Project Structure

```
├── data_for_ml.csv               # Dataset
├── model1_hours_studied.py       # Model 1 — Hours Studied
├── model2_sleep_hours.py         # Model 2 — Sleep Hours
├── model3_internet_usage.py      # Model 3 — Internet Usage
├── model4_attendance.py          # Model 4 — Attendance
├── model5_gender.py              # Model 5 — Gender
└── README.md
```

---

## 📊 Dataset

| Column | Description |
|--------|-------------|
| `Hours_Studied` | Roz kitne ghante parhai ki |
| `Sleep_Hours` | Roz kitne ghante soye |
| `Internet_Usage` | Roz kitne ghante internet use kiya |
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

## ⚙️ Har Model ka Flow

Har `.py` file mein yeh steps hain:

1. **Data Load** — CSV file se data load karo
2. **Cleaning** — NaN values fill ya drop karo
3. **X aur y** — Input aur output alag karo
4. **Train Test Split** — 70% train, 30% test
5. **StandardScaler** — Data scale karo
6. **LinearRegression** — Model train karo
7. **R2 Score** — Model evaluate karo
8. **joblib.dump** — Model save karo
9. **joblib.load** — Model dobara load karo
10. **Predict Function** — `input()` se value lo aur predict karo

---

## 🚀 How to Run

### Requirements install karo:
```bash
pip install pandas scikit-learn joblib
```

### Koi bhi model chalao:
```bash
python model1_hours_studied.py
python model2_sleep_hours.py
python model3_internet_usage.py
python model4_attendance.py
python model5_gender.py
```

### Input example:
```
Enter Hours Studied: 7
Predicted Final Score: 76.43
```

---

## 📦 Requirements

```
pandas
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

- Dataset mein kuch missing values aur typos thay (`Femle`, `male`) jo automatically fix ho jaate hain
- `Gender` column ko `LabelEncoder` se numeric banaya gaya hai (Model 5)
- Sab models `StandardScaler` use karte hain
- Saved model files same folder mein store hongi
