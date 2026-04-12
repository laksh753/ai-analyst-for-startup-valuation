# 🚀 Startup Success Prediction using Machine Learning

![Dashboard Preview](startup_success_dashboard.png)

> Predict whether a startup will **Fail, Get Acquired, or Achieve IPO** using early-stage metrics like funding, revenue, and team data.

---

## 📌 Project Overview

This project builds a **multi-class machine learning model** to predict startup outcomes using **100,000 real-world-like records**.

Unlike traditional binary success/failure models, this system predicts:
- ✅ Acquisition  
- ❌ Failure  
- 🚀 IPO  

---

## 📊 Dataset Summary

- **Total Records:** 100,000  
- **Features:** 10 input features  
- **Target:** `outcome` (3 classes)

### 📈 Class Distribution:
- ❌ Failure → **55,610 (55.6%)**
- ✅ Acquisition → **42,335 (42.3%)**
- 🚀 IPO → **2,055 (2.1%)**

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:**
  - pandas, numpy
  - scikit-learn
  - matplotlib, seaborn

---

## 🧠 Machine Learning Models

| Model                  | Accuracy |
|----------------------|---------|
| Logistic Regression  | 65.20%  |
| Random Forest 🌲     | **73.81%** |

🏆 **Best Model: Random Forest**

---

## 📊 Model Performance (Random Forest)

- **Accuracy:** 73.81%  
- **Precision:** 0.74  
- **Recall:** 0.74  
- **F1 Score:** 0.73  

---

## 🔍 Confusion Matrix (Random Forest)

| Actual \ Predicted | Acquisition | Failure | IPO |
|------------------|------------|---------|-----|
| Acquisition       | 5544       | 2883    | 40  |
| Failure           | 1984       | 9138    | 0   |
| IPO               | 330        | 1       | 80  |

---

## 🌲 Feature Importance (Top Drivers)

| Rank | Feature                    | Importance |
|------|--------------------------|------------|
| 1️⃣   | Revenue ($M)             | 0.2899     |
| 2️⃣   | Product Traction (Users) | 0.1968     |
| 3️⃣   | Founder Experience       | 0.0987     |

---

## 🧪 Machine Learning Pipeline

1. Data Loading  
2. Data Cleaning  
3. Encoding  
4. Train-Test Split (80/20)  
5. Model Training  
6. Evaluation  

---

## 📁 Project Structure

```
Startup-Success-Prediction
│
├── startup_success_ml.py
├── startup_success_dashboard.png
├── startup_success_ppt.html
├── dataset.csv (not included)
└── README.md
```

---

## 💡 Key Findings

- Revenue is the strongest predictor  
- Most startups fail (~56%)  
- Random Forest performs best  
- IPO prediction is hardest  

---

## ⚠️ Limitations

- Class imbalance (IPO = 2.1%)  
- Synthetic dataset  
- No time-based validation  

---

## 🚀 Future Improvements

- SMOTE for imbalance  
- XGBoost / LightGBM  
- Feature engineering  
- Deployment with FastAPI  

---

## 👨‍💻 Author

Laksh Arora  
Manipal University Jaipur  
PBL Project 2026  

---

## ⭐ How to Run

```
pip install pandas numpy matplotlib seaborn scikit-learn
python startup_success_ml.py
```

---

## 📌 Conclusion

Machine learning can help predict startup success and support better investment decisions.
