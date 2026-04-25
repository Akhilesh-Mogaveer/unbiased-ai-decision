# ⚖️ Unbiased AI Decision
### Ensuring Fairness and Detecting Bias in Automated Decisions

> Built for **Hack2Skill Solution Challenge 2026 — Build with AI**
> Powered by Gemini API · Firebase · fairlearn · scikit-learn

[![Live Demo](https://img.shields.io/badge/Live%20Demo-unbiased--ai--audit.web.app-7b6cff?style=for-the-badge)](https://unbiased-ai-audit.web.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Firebase](https://img.shields.io/badge/Firebase-Hosting-orange?style=for-the-badge&logo=firebase)](https://firebase.google.com)
[![Gemini](https://img.shields.io/badge/Gemini-API-4285F4?style=for-the-badge&logo=google)](https://ai.google.dev)

---

## 🚨 The Problem

Computer programs now make life-changing decisions — who gets hired, who receives a loan, who gets medical care. When these models learn from flawed historical data, they silently repeat and amplify discrimination at scale.

- Amazon's AI hiring tool penalized resumes containing the word "women's"
- US healthcare algorithms systematically under-treated Black patients
- Facial recognition systems misidentify darker-skinned individuals at 35x the error rate of lighter-skinned individuals

**Organizations have no easy way to detect this bias before deployment.**

---

## ✅ Our Solution

**Unbiased AI Decision** is a bias audit tool that lets any organization upload their dataset and ML model to receive a full fairness report — in seconds.

It measures **9 industry-standard fairness metrics**, flags violations, explains findings in plain English using Gemini AI, and recommends specific fixes.

---

## 🎯 Key Features

- **Data-Level Audit** — detects representation imbalance and outcome rate gaps before model training
- **9 Fairness Metrics** — Demographic Parity, Equalized Odds, Disparate Impact (legal 4/5ths rule), Equal Opportunity, and Intersectional Analysis
- **Gemini AI Explanation** — converts technical bias findings into plain English for non-technical decision makers
- **Fix Recommendations** — specific, actionable fixes with library references (aif360, fairlearn, SMOTE)
- **Interactive Dashboard** — dark-themed web interface with live Chart.js visualizations
- **CSV Upload** — audit any HR or decision-making dataset, not just IBM HR

---

## 📊 Real Findings on IBM HR Dataset

| Protected Attribute | Metric | Value | Status |
|---|---|---|---|
| Gender | Demographic Parity | 0.0379 | ✅ PASS |
| Age Group | Demographic Parity | 1.0 | ❌ FAIL |
| Gender | Disparate Impact | 0.3724 | ❌ FAIL |
| Age Group | Disparate Impact | 0.0 | ❌ FAIL |
| Marital Status | Disparate Impact | 0.0 | ❌ FAIL |
| Age Group | Equal Opportunity | 0.3333 | ❌ FAIL |
| Gender | Equalized Odds | 0.05 | ✅ PASS |
| Age Group | Equalized Odds | 1.0 | ❌ FAIL |

**Model Accuracy: 84.4% · 6 Failing · 3 Passing · Biggest Bias: Marital Status (15.4% gap)**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Hosting | Firebase Hosting |
| Database | Cloud Firestore |
| AI Explanation | Gemini 2.0 Flash API |
| Backend API | Python Flask + Cloud Run |
| ML Model | scikit-learn (Random Forest + Logistic Regression) |
| Fairness Metrics | fairlearn, aif360 |
| Data Processing | pandas, numpy |

---

## 📁 Project Structure

```
unbiased-ai-decision/
├── notebooks/
│   ├── data_audit.ipynb          # Phase 1: data-level bias analysis
│   └── model_training.ipynb      # Phase 2: model training + 9 fairness metrics
├── backend/
│   └── app.py                    # Flask API (3 endpoints)
├── frontend/
│   └── index.html                # Complete dashboard UI
├── data/
│   └── ibm_hr_with_agegroup.csv  # Cleaned dataset
├── models/
│   ├── random_forest.pkl         # Trained RF model
│   └── logistic_regression.pkl   # Trained LR model
├── reports/
│   ├── bias_report.json          # Full audit report
│   └── charts/                   # 4 analysis charts (PNG)
├── .env                          # API keys (not committed)
├── firebase.json
└── README.md
```

---

## 🚀 Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Akhilesh-Mogaveer/unbiased-ai-decision.git
cd unbiased-ai-decision
```

### 2. Install Python dependencies
```bash
pip install pandas numpy scikit-learn fairlearn aif360 matplotlib seaborn flask flask-cors google-generativeai firebase-admin python-dotenv joblib
```

### 3. Add your Gemini API key
Create a `.env` file in the root folder:
```
GEMINI_API_KEY=your_gemini_api_key_here
```
Get a free key at [aistudio.google.com](https://aistudio.google.com)

### 4. Run the notebooks
Open and run all cells in order:
```
notebooks/data_audit.ipynb       → generates charts + cleaned dataset
notebooks/model_training.ipynb   → trains model + generates bias_report.json
```

### 5. Start the Flask API
```bash
cd backend
python app.py
```
API runs at `http://localhost:5000`

### 6. Open the dashboard
Open `frontend/index.html` in your browser and click **Run Demo Audit**.

---

## 🌐 Live Demo

**[https://unbiased-ai-audit.web.app](https://unbiased-ai-audit.web.app)**

The live site is hosted on Firebase Hosting. Click **Run Demo Audit** to see the full bias report rendered from real model outputs.

---

## 📐 Fairness Metrics Explained

| Metric | What It Measures | Threshold |
|---|---|---|
| **Demographic Parity** | Gap in prediction rates between groups | < 0.1 |
| **Equalized Odds** | Gap in both TPR and FPR between groups | < 0.1 |
| **Disparate Impact** | Ratio of selection rates (legal 4/5ths rule) | ≥ 0.8 |
| **Equal Opportunity** | Gap in True Positive Rate (recall) between groups | < 0.1 |
| **Intersectional Analysis** | Bias across combined attribute slices (e.g. Gender × Age) | < 0.1 |

---

## 🔧 Fix Recommendations Generated

1. **Age Group Bias** → Apply `aif360.algorithms.preprocessing.Reweighing`
2. **Marital Status Disparate Impact** → Apply `fairlearn.postprocessing.ThresholdOptimizer`
3. **Gender Parity Borderline** → Apply `imblearn.over_sampling.SMOTE`
4. **Proxy Feature Leakage** → Use `sklearn.inspection.permutation_importance`


## 📄 License

MIT License — free to use, modify, and distribute.
