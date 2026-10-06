# User Conversion Prediction – Stacking Ensemble Pipeline

A machine learning pipeline that predicts whether a website user will convert (`Converted` = 0 or 1). It uses leak-free target encoding, engineered features, a stacked ensemble of gradient boosting models, Optuna tuning, and F1-optimized decision thresholding.

## Dataset

The project uses a competition-style split:

- `train.csv` – labeled training data
- `public_test.csv` – public leaderboard set (also merged into training)
- `private_test.csv` – final test set used for `submission.csv`

The data files are not included in this repo. Place them in the same folder as `train_pipeline.py`.

**Columns:** `User_ID`, `Age`, `Income`, `Pages_Viewed`, `Products_Viewed`, `Time_On_Site`, `Previous_Purchases`, `Discount_Seen`, `Device_Type`, `Traffic_Source`, `Campaign_Code`, `Browser_Version`, `City_Tier`

**Target:** `Converted` (binary)

## Approach

1. **Missing values:** binary missing-flags for `Age`, `Income` and `Time_On_Site`, then median imputation.
2. **Feature engineering:** 20+ ratio, interaction and non-linear features, such as `product_page_ratio`, `intent_score`, `session_depth`, `loyalty_score`, `affordability`, `discount_interest` and `engagement_composite`.
3. **Leak-free encoding:** 5-fold stratified out-of-fold target encoding for categorical columns, plus frequency encoding for high-cardinality columns (`Campaign_Code`, `Browser_Version`).
4. **Models:** CatBoost, LightGBM and XGBoost, validated with stratified 5-fold cross-validation. ExtraTrees is used as a baseline comparison.
5. **Tuning:** Optuna hyperparameter search.
6. **Ensemble:** weights and decision threshold are searched jointly to maximize F1.
7. **Output:** final predictions for the private test set are written to `submission.csv`.

## EDA notes

- Target imbalance: about 31% converted, 69% not converted
- Strongest correlations with the target: `Pages_Viewed` and `Products_Viewed` (about 0.31 each), then `Discount_Seen` (about 0.11)
- Missing values: `Age`, `Income` and `Time_On_Site`
- `Campaign_Code` has about 9,000 unique values, so it needs special encoding
- Referral traffic converts best; paid ads convert lowest

## Files

```
train_pipeline.py   standalone pipeline script
notebook.ipynb      step-by-step notebook with EDA
submission.csv      final predictions
README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn catboost lightgbm xgboost optuna matplotlib seaborn
python train_pipeline.py
```

Or open `notebook.ipynb` in Jupyter.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, CatBoost, LightGBM, XGBoost, Optuna
