# 🚀 AI Startup Profitability Predictor
### PBL Project · Manipal University Jaipur · 2026
### Department of Computer Science & Engineering

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Project Overview

Can publicly available funding metadata predict whether an AI startup will become profitable — *before* market outcomes are known?

This project builds and evaluates two binary classification models on a curated dataset of 60 global AI startups (2020–2023). Using features like funding stage, valuation, employee count, and AI model type, we predict whether a startup achieves profitability.

> **Best Result:** Logistic Regression achieved **66.7% accuracy** on the held-out test set.

---

## 🗂️ Repository Structure

```
ai-startup-profitability-predictor/
│
├── ai_startup_funding.csv          # Raw dataset (60 AI startup records)
├── ai_startup_ml.py                # Full ML pipeline (beginner-friendly, heavily commented)
├── ai_startup_ml_dashboard.png     # Output visualisation dashboard
├── pbl_ppt_final.html              # Interactive PBL presentation
└── README.md                       # This file
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Records | 60 AI startups |
| Time Period | 2020 – 2023 |
| Countries | USA, UK, Canada, Germany, France, Israel, Australia, Hong Kong |
| Target Variable | `Profitability` (1 = Profitable, 0 = Not Profitable) |
| Class Balance | 22 Profitable (37%) · 38 Not Profitable (63%) |

### Features Used

| Feature | Type | Description |
|---|---|---|
| `Industry_AI_Application` | Categorical | Sector the startup operates in |
| `Country` | Categorical | Country of headquarters |
| `Funding_Stage` | Categorical | Series A/B/C/D/E/F, IPO, Acquired |
| `Valuation ($M)` | Numerical | Company valuation in millions USD |
| `Funding_Amount ($M)` | Numerical | Total funding raised in millions USD |
| `Year` | Numerical | Year of the funding round |
| `Employee_Count` | Numerical | Number of full-time employees |
| `AI_Model_Type` | Categorical | Core AI technology used |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-startup-profitability-predictor.git
cd ai-startup-profitability-predictor
```

### 2. Set up a virtual environment
```bash
python -m venv ml_env
source ml_env/bin/activate        # Mac / Linux
ml_env\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Run the ML pipeline
```bash
python ai_startup_ml.py
```

The script will print all results to the terminal and save `ai_startup_ml_dashboard.png` in the same folder.

---

## 🤖 Models

### Model 1 — Logistic Regression 🏆
- `max_iter = 1000`
- L2 regularisation
- `random_state = 42`
- **Accuracy: 66.7%**

### Model 2 — Random Forest Classifier
- `n_estimators = 100` (100 decision trees)
- `max_depth = 5`
- Gini criterion
- `random_state = 42`
- **Accuracy: 50.0%**

---

## 📈 Results

### Accuracy Comparison

| Model | Accuracy | Precision (avg) | Recall (avg) | F1-Score (avg) |
|---|---|---|---|---|
| **Logistic Regression** | **66.7%** | 0.63 | 0.67 | 0.63 |
| Random Forest | 50.0% | 0.50 | 0.50 | 0.50 |

### Logistic Regression — Confusion Matrix

```
                  Predicted: No    Predicted: Yes
Actual: No              7               1
Actual: Yes             3               1
```

### 🌲 Feature Importance (Random Forest)

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | AI Model Type | 0.1822 |
| 2 | Industry / AI Application | 0.1669 |
| 3 | Funding Stage | 0.1515 |
| 4 | Valuation ($M) | 0.1272 |
| 5 | Funding Amount ($M) | 0.1172 |
| 6 | Employee Count | 0.1066 |
| 7 | Year | 0.0872 |
| 8 | Country | 0.0612 |

---

## 💡 Key Findings

- **AI Model Type** is the strongest predictor of profitability — the specific technology a startup uses matters more than how much money it raised.
- **Logistic Regression outperformed Random Forest** on this dataset. With only 60 rows, simpler models generalise better than complex ensemble methods.
- The model has **high recall (0.88) for the "Not Profitable" class** — it is good at flagging startups likely to fail, which is the more valuable signal for investors.

---

## ⚠️ Limitations

- Small dataset (60 rows) limits statistical confidence and generalisation
- Class imbalance (63% not profitable) skews predictions toward the majority class
- LabelEncoding imposes artificial ordinal relationships on nominal categories
- No temporal train/test split — data leakage from future rounds is possible

---

## 🔭 Future Scope

- [ ] Expand dataset to 500+ records using the Crunchbase API
- [ ] Apply **SMOTE** to handle class imbalance
- [ ] Implement **cross-validation** for more robust evaluation
- [ ] Try **XGBoost / LightGBM** for better non-linear pattern capture
- [ ] Add NLP features extracted from investor pitch descriptions
- [ ] Deploy as a web app using **FastAPI + React**

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| pandas | Data loading, cleaning, manipulation |
| numpy | Numerical operations |
| scikit-learn | ML models, encoding, evaluation |
| matplotlib | Chart generation |
| seaborn | Statistical visualisations |

---

## 👤 Academic Credits

| Role | Name | Registration No. |
|---|---|---|
| **Project Guide** | Dr. Amit Kumar Gupta | — |
| **Student** | Laksh Arora | 2427030030 |

**Institution:** Manipal University Jaipur  
**Department:** Computer Science & Engineering  
**Academic Year:** 2025–2026

---

## 📄 License

This project is licensed under the MIT License — feel free to use and modify it for educational purposes.

---

*Built with ❤️ for PBL 2026 · Manipal University Jaipur*
