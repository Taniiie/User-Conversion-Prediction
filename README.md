# 🏆 Nutrition Health Survey - Age Prediction & Conversion ML Pipeline

An end-to-end, high-performance machine learning pipeline engineered for top-tier competitive performance on the **Nutrition Health Survey (NHANES)** dataset. Features leak-free out-of-fold target encoding, 20+ domain-engineered interaction features, multi-model probability stacking (CatBoost, LightGBM, XGBoost, ExtraTrees), Optuna hyperparameter tuning, and decision threshold optimization.

---

## 📊 Dataset Overview

The dataset is derived from the **National Health and Nutrition Examination Survey (NHANES)**, a comprehensive program of studies designed to assess the health and nutritional status of adults and children in the United States.

### 📁 Dataset Structure & Split
- **Training Set (`train.csv`)**: Baseline labeled dataset containing patient demographics, clinical markers, and target labels.
- **Public Test Set (`public_test.csv`)**: Evaluated on the public leaderboard.
- **Private Test Set (`private_test.csv`)**: Used for final leaderboard ranking.

### 🔬 Feature Dictionary & Domain Categories

| Category | Feature Name | Description / Data Type | Preprocessing Strategy |
| :--- | :--- | :--- | :--- |
| **Demographics** | `Age` | Patient age in years | Median Imputed + Missing Indicator (`Age_missing`) |
| **Financial / Status** | `Income` | Family income ratio / bracket | Median Imputed + Log Transform (`log_income`) |
| **Engagement** | `Pages_Viewed` | Number of health survey pages viewed | Polynomial Features (`pages_sq`) |
| **Engagement** | `Products_Viewed` | Specific health items / modules viewed | Ratio Features (`product_page_ratio`) |
| **Time & Duration** | `Time_On_Site` | Session duration in minutes | Log Transform + Clipped at 60m |
| **User History** | `Previous_Purchases` | Prior survey participation count | Loyalty Interaction (`loyalty_score`) |
| **Promotion** | `Discount_Seen` | Flag for health incentive offers | Discount Interaction Scores |
| **Categorical** | `Device_Type` | Access device (Mobile/Desktop/Tablet) | Out-Of-Fold Stratified Target Encoding |
| **Categorical** | `Traffic_Source` | Referral path / Direct / Search | Out-Of-Fold Stratified Target Encoding |
| **Categorical** | `Campaign_Code` | High-cardinality campaign code | KFold Target Encoding + Frequency Encoding |
| **Categorical** | `Browser_Version` | Browser version identifier | Target & Frequency Encoding |
| **Categorical** | `City_Tier` | Geographic tier classification | Target Encoding |
| **Target Variable** | `Converted` / `Age_Group` | Binary Classification Target (`0` or `1`) | Stratified Split Target |

---

## 📌 Key Pipeline Highlights

- **🔍 Advanced Missing Value Engineering**: Binary missingness indicators (`Age_missing`, `Income_missing`, `Time_missing`, `total_missing`) + leak-free median imputation.
- **⚡ 20+ Domain Feature Engineering**: 
  - *Engagement Ratios*: `product_page_ratio`, `intent_score`, `session_depth`, `browsing_intensity`, `product_intensity`.
  - *Financial & Demographics*: `loyalty_score`, `income_purchase_ratio`, `affordability`, `age_income`, `browser_age`.
  - *Discount & Composites*: `discount_interest`, `discount_intent`, `engagement_composite`, `returning_buyer`.
- **🛡️ Leak-Free Target Encoding**: Stratified 5-Fold KFold target encoding for categorical features preventing data leakage.
- **🤖 Multi-Model Ensemble**:
  - CatBoost Classifier
  - LightGBM Classifier
  - XGBoost Classifier
  - ExtraTrees Classifier
- **🎯 Hyperparameter & Threshold Optimization**: Automated Optuna Bayesian tuning and joint grid-search optimization for ensemble weights and decision threshold $t$ maximizing F1 Score.

---

## 📁 Project Structure

```
d:\Nutrition Health Survey\
├── notebook.ipynb            # Full Interactive EDA & Competition Jupyter Notebook
├── train_pipeline.py         # Modular Standalone Python Pipeline Script
├── submission.csv            # Final Optimized Competition Submission
├── methodology_report (1).pdf # Technical Project Documentation
└── README.md                 # Project Overview & Dataset Documentation
```

---

## 🚀 Getting Started

### 1. Installation & Setup

Ensure Python 3.9+ and required libraries are installed:
```bash
pip install pandas numpy scikit-learn catboost lightgbm xgboost optuna matplotlib seaborn
```

### 2. Running the Pipeline

Run the complete pipeline from terminal:
```bash
python train_pipeline.py
```

Or open `notebook.ipynb` in Jupyter Notebook / VS Code for step-by-step interactive visualizations.

---

## 📊 Pipeline Methodology

```
┌─────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐
│  Data Ingestion │ ──► │ Feature Engineering   │ ──► │ Leak-Free Target       │
│  & Missing Flags│     │ (20+ Ratios & Logs)   │     │ Encoding (5-Fold KFold)│
└─────────────────┘     └───────────────────────┘     └────────────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐
│ Final Export    │ ◄── │ Weight & Decision     │ ◄── │ Stratified CV &        │
│ submission.csv  │     │ Threshold Tuning (F1) │     │ Optuna Hyper-Tuning    │
└─────────────────┘     └───────────────────────┘     └────────────────────────┘
```

---

## 📜 License & Acknowledgments

Built for CDC NHANES / Nutrition Health Survey data modeling. All code follows strictly leak-free validation protocols.
