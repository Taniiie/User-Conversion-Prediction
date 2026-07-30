"""
Full Machine Learning Pipeline for Binary Classification (Top-1 Competition Strategy)
Folder: Nutrition Health Survey

Pipeline Workflow:
1. Data Loading & Exploratory Preprocessing
2. Missing Value Analysis & Indicator Engineering
3. Advanced Domain Feature Engineering (20+ Ratios, Interactions & Transformations)
4. Out-Of-Fold KFold Target Encoding (Leak-Free)
5. Frequency Encoding for High-Cardinality Variables
6. 5-Fold Stratified Cross-Validation across 4 Diverse Models (CatBoost, LightGBM, XGBoost, ExtraTrees)
7. Optuna Hyperparameter Optimization (Optimizing F1 Score)
8. OOF Probability Stacking & Prediction Generation
9. Joint Ensemble Weighting & Decision Threshold Optimization
10. Submission CSV Export
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


# ==========================================
# 1. Feature Engineering & Preprocessing
# ==========================================

def add_missing_indicators(df):
    """Creates binary missingness indicator flags for numerical columns."""
    df = df.copy()
    if 'Age' in df.columns:
        df['Age_missing'] = df['Age'].isnull().astype(int)
    if 'Income' in df.columns:
        df['Income_missing'] = df['Income'].isnull().astype(int)
    if 'Time_On_Site' in df.columns:
        df['Time_missing'] = df['Time_On_Site'].isnull().astype(int)
    
    missing_cols = [c for c in ['Age_missing', 'Income_missing', 'Time_missing'] if c in df.columns]
    if missing_cols:
        df['total_missing'] = df[missing_cols].sum(axis=1)
    return df


def engineer_features(df):
    """Engineers 20+ domain-specific interaction, ratio, and non-linear features."""
    df = df.copy()
    
    # --- Engagement Ratios & Intensity ---
    if 'Products_Viewed' in df.columns and 'Pages_Viewed' in df.columns:
        df['product_page_ratio'] = df['Products_Viewed'] / (df['Pages_Viewed'] + 1)
        df['pages_minus_products'] = df['Pages_Viewed'] - df['Products_Viewed']
        df['products_sq'] = df['Products_Viewed'] ** 2
        df['pages_sq'] = df['Pages_Viewed'] ** 2
        
    if 'Products_Viewed' in df.columns and 'Time_On_Site' in df.columns:
        df['intent_score'] = df['Products_Viewed'] * df['Time_On_Site']
        df['product_intensity'] = df['Products_Viewed'] / (df['Time_On_Site'] + 0.1)
        
    if 'Pages_Viewed' in df.columns and 'Time_On_Site' in df.columns:
        df['session_depth'] = df['Pages_Viewed'] * df['Time_On_Site']
        df['browsing_intensity'] = df['Pages_Viewed'] / (df['Time_On_Site'] + 0.1)
        
    # --- Financial & Demographic Interactions ---
    if 'Previous_Purchases' in df.columns and 'Income' in df.columns:
        df['loyalty_score'] = df['Previous_Purchases'] * df['Income']
        df['income_purchase_ratio'] = df['Income'] / (df['Previous_Purchases'] + 1)
        
    if 'Income' in df.columns and 'Age' in df.columns:
        df['affordability'] = df['Income'] / (df['Age'] + 1)
        df['age_income'] = df['Age'] * df['Income']
        
    if 'Browser_Version' in df.columns and 'Age' in df.columns:
        df['browser_age'] = df['Browser_Version'] * df['Age']
        
    # --- Discount Interaction Features ---
    if 'Discount_Seen' in df.columns:
        if 'Products_Viewed' in df.columns:
            df['discount_interest'] = df['Discount_Seen'] * df['Products_Viewed']
        if 'intent_score' in df.columns:
            df['discount_intent'] = df['Discount_Seen'] * df['intent_score']
            
    # --- Non-linear Transformations ---
    if 'Income' in df.columns:
        df['log_income'] = np.log1p(np.maximum(0, df['Income']))
    if 'Time_On_Site' in df.columns:
        df['log_time'] = np.log1p(np.maximum(0, df['Time_On_Site']))
        df['time_clipped'] = df['Time_On_Site'].clip(upper=60)
        
    # --- Behavioral & Segmentation Flags ---
    if 'Pages_Viewed' in df.columns and 'Products_Viewed' in df.columns:
        df['high_engagement'] = ((df['Pages_Viewed'] > 20) & (df['Products_Viewed'] > 15)).astype(int)
    if 'Previous_Purchases' in df.columns:
        df['returning_buyer'] = (df['Previous_Purchases'] > 0).astype(int)
        
    # --- Composite Score ---
    req_cols = ['Pages_Viewed', 'Products_Viewed', 'Time_On_Site', 'Discount_Seen']
    if all(c in df.columns for c in req_cols):
        df['engagement_composite'] = (
            df['Pages_Viewed'] * 0.3 +
            df['Products_Viewed'] * 0.4 +
            df['Time_On_Site'] * 0.1 +
            df['Discount_Seen'] * 0.2
        )
        
    return df


def kfold_target_encode(train_df, test_df, cols, target_col, n_splits=5, seed=42):
    """Leak-free Out-Of-Fold Target Encoding using Stratified K-Fold."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for col in cols:
        if col not in train_df.columns:
            continue
        enc_col = f'{col}_te'
        train_df[enc_col] = 0.0
        
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_df, train_df[target_col])):
            means = train_df.iloc[tr_idx].groupby(col)[target_col].mean()
            train_df.loc[train_df.index[val_idx], enc_col] = train_df.iloc[val_idx][col].map(means)
            
        global_mean = train_df[target_col].mean()
        train_df[enc_col] = train_df[enc_col].fillna(global_mean)
        
        if test_df is not None and col in test_df.columns:
            full_means = train_df.groupby(col)[target_col].mean()
            test_df[enc_col] = test_df[col].map(full_means).fillna(global_mean)
            
    return train_df, test_df


# ==========================================
# 2. Model Evaluation Helper
# ==========================================

def evaluate_model(model_fn, X, y, model_name, skf):
    """Evaluates model using Stratified K-Fold CV and collects OOF probability predictions."""
    oof_preds = np.zeros(len(y))
    fold_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = model_fn()
        
        if model_name == 'CatBoost':
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        elif model_name == 'LightGBM':
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        elif model_name == 'XGBoost':
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        else:
            model.fit(X_tr, y_tr)
            
        if hasattr(model, 'predict_proba'):
            val_proba = model.predict_proba(X_val)[:, 1]
        else:
            val_proba = model.predict(X_val)
            
        oof_preds[val_idx] = val_proba
        val_pred = (val_proba >= 0.5).astype(int)
        score = f1_score(y_val, val_pred)
        fold_scores.append(score)
        print(f"   Fold {fold+1}: F1 = {score:.5f}")
        
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    print(f" >> {model_name} Baseline CV F1: {mean_f1:.5f} (+/- {std_f1:.5f})\n")
    return oof_preds, mean_f1


# ==========================================
# 3. Main Execution Function
# ==========================================

def run_pipeline():
    print("=" * 60)
    print("STARTING FULL ML BINARY CLASSIFICATION PIPELINE")
    print("=" * 60)
    
    # --- Load Datasets ---
    data_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(data_dir, 'train.csv')
    public_test_path = os.path.join(data_dir, 'public_test.csv')
    private_test_path = os.path.join(data_dir, 'private_test.csv')
    
    if not os.path.exists(train_path):
        print(f"Warning: '{train_path}' not found. Please place dataset CSV files in the project directory.")
        return
        
    train = pd.read_csv(train_path)
    public_test = pd.read_csv(public_test_path) if os.path.exists(public_test_path) else None
    private_test = pd.read_csv(private_test_path) if os.path.exists(private_test_path) else None
    
    TARGET = 'Converted'
    
    if public_test is not None:
        full_train = pd.concat([train, public_test], axis=0, ignore_index=True)
    else:
        full_train = train.copy()
        
    print(f"Full Train Shape: {full_train.shape}")
    if private_test is not None:
        print(f"Private Test Shape: {private_test.shape}")
        
    # --- Missing Indicators & Imputation ---
    print("\n--- Phase 2: Missing Value Engineering ---")
    full_train = add_missing_indicators(full_train)
    if private_test is not None:
        private_test = add_missing_indicators(private_test)
        
    num_cols_to_fill = ['Age', 'Income', 'Time_On_Site']
    for col in num_cols_to_fill:
        if col in full_train.columns:
            med = train[col].median()
            full_train[col] = full_train[col].fillna(med)
            if private_test is not None and col in private_test.columns:
                private_test[col] = private_test[col].fillna(med)
                
    # --- Feature Engineering ---
    print("\n--- Phase 3: Advanced Feature Engineering ---")
    full_train = engineer_features(full_train)
    if private_test is not None:
        private_test = engineer_features(private_test)
        
    # --- Target Encoding ---
    print("\n--- Phase 4: KFold Out-Of-Fold Target Encoding ---")
    target_encode_cols = ['Device_Type', 'Traffic_Source', 'Campaign_Code', 'Browser_Version', 'City_Tier']
    full_train, private_test = kfold_target_encode(full_train, private_test, target_encode_cols, TARGET)
    
    # --- Frequency Encoding ---
    print("\n--- Phase 5: Frequency Encoding ---")
    freq_cols = ['Campaign_Code', 'Browser_Version']
    for col in freq_cols:
        if col in full_train.columns:
            freq = full_train[col].value_counts()
            full_train[f'{col}_freq'] = full_train[col].map(freq)
            if private_test is not None and col in private_test.columns:
                private_test[f'{col}_freq'] = private_test[col].map(freq).fillna(0)
                
    # --- Feature Matrix Construction ---
    drop_cols = ['User_ID', TARGET, 'Device_Type', 'Traffic_Source']
    feature_cols = [c for c in full_train.columns if c not in drop_cols]
    
    X_full = full_train[feature_cols].values
    y_full = full_train[TARGET].values
    
    print(f"\nFinal Feature Count: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    
    # --- Stratified KFold Setup ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # --- Model Evaluation (Baselines) ---
    print("\n--- Phase 6: Baseline Model Training (5-Fold CV) ---")
    
    print("1. CatBoost...")
    cat_oof, cat_cv = evaluate_model(
        lambda: CatBoostClassifier(
            iterations=1500, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, bagging_temperature=0.8, random_strength=1.0,
            loss_function='Logloss', eval_metric='F1',
            random_seed=42, verbose=0, early_stopping_rounds=100
        ),
        X_full, y_full, 'CatBoost', skf
    )
    
    print("2. LightGBM...")
    lgb_oof, lgb_cv = evaluate_model(
        lambda: LGBMClassifier(
            n_estimators=1500, max_depth=6, learning_rate=0.05,
            num_leaves=40, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, min_child_samples=20,
            objective='binary', metric='binary_logloss',
            random_state=42, verbose=-1, n_jobs=-1
        ),
        X_full, y_full, 'LightGBM', skf
    )
    
    print("3. XGBoost...")
    xgb_oof, xgb_cv = evaluate_model(
        lambda: XGBClassifier(
            n_estimators=1500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42,
            verbosity=0, early_stopping_rounds=100, n_jobs=-1
        ),
        X_full, y_full, 'XGBoost', skf
    )
    
    print("4. ExtraTrees...")
    et_oof, et_cv = evaluate_model(
        lambda: ExtraTreesClassifier(
            n_estimators=500, max_depth=12,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        X_full, y_full, 'ExtraTrees', skf
    )
    
    # --- Optuna Tuning ---
    print("\n--- Phase 7: Optuna Hyperparameter Tuning ---")
    
    def optuna_catboost(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 2.0),
            'loss_function': 'Logloss', 'eval_metric': 'F1',
            'random_seed': 42, 'verbose': 0, 'early_stopping_rounds': 100,
        }
        scores = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
            model = CatBoostClassifier(**params)
            model.fit(X_full[tr_idx], y_full[tr_idx], eval_set=(X_full[val_idx], y_full[val_idx]), verbose=0)
            val_proba = model.predict_proba(X_full[val_idx])[:, 1]
            best_f1 = max(f1_score(y_full[val_idx], (val_proba >= t).astype(int)) for t in np.arange(0.30, 0.60, 0.02))
            scores.append(best_f1)
        return np.mean(scores)

    print("Tuning CatBoost (15 trials)...")
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(optuna_catboost, n_trials=15)
    print(f"Best CatBoost CV F1: {study_cat.best_value:.5f}")

    def optuna_lgbm(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'objective': 'binary', 'metric': 'binary_logloss',
            'random_state': 42, 'verbose': -1, 'n_jobs': -1,
        }
        scores = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
            model = LGBMClassifier(**params)
            model.fit(X_full[tr_idx], y_full[tr_idx], eval_set=[(X_full[val_idx], y_full[val_idx])])
            val_proba = model.predict_proba(X_full[val_idx])[:, 1]
            best_f1 = max(f1_score(y_full[val_idx], (val_proba >= t).astype(int)) for t in np.arange(0.30, 0.60, 0.02))
            scores.append(best_f1)
        return np.mean(scores)

    print("Tuning LightGBM (15 trials)...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(optuna_lgbm, n_trials=15)
    print(f"Best LightGBM CV F1: {study_lgb.best_value:.5f}")
    
    # --- Retrain Best Models & Predict on Test ---
    if private_test is not None:
        X_private = private_test[[c for c in feature_cols if c in private_test.columns]].values
        
        print("\n--- Retraining Tuned Models & Generating Test Predictions ---")
        
        def train_and_predict_test(model_class, best_params, model_name):
            oof = np.zeros(len(y_full))
            test_preds = np.zeros(len(X_private))
            
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
                model = model_class(**best_params)
                if model_name == 'CatBoost':
                    model.fit(X_full[tr_idx], y_full[tr_idx], eval_set=(X_full[val_idx], y_full[val_idx]), verbose=0)
                elif model_name == 'LightGBM':
                    model.fit(X_full[tr_idx], y_full[tr_idx], eval_set=[(X_full[val_idx], y_full[val_idx])])
                else:
                    model.fit(X_full[tr_idx], y_full[tr_idx])
                    
                oof[val_idx] = model.predict_proba(X_full[val_idx])[:, 1]
                test_preds += model.predict_proba(X_private)[:, 1] / skf.n_splits
                
            return oof, test_preds
            
        cat_params = study_cat.best_params.copy()
        cat_params.update({'loss_function': 'Logloss', 'eval_metric': 'F1', 'random_seed': 42, 'verbose': 0})
        cat_oof_t, cat_test = train_and_predict_test(CatBoostClassifier, cat_params, 'CatBoost')
        
        lgb_params = study_lgb.best_params.copy()
        lgb_params.update({'objective': 'binary', 'metric': 'binary_logloss', 'random_state': 42, 'verbose': -1, 'n_jobs': -1})
        lgb_oof_t, lgb_test = train_and_predict_test(LGBMClassifier, lgb_params, 'LightGBM')
        
        # --- Threshold & Weight Optimization ---
        print("\n--- Optimizing Ensemble Weights & Decision Threshold ---")
        best_ensemble_f1 = 0
        best_weights = (0.5, 0.5)
        best_threshold = 0.5
        
        for w1 in np.arange(0.1, 0.9, 0.05):
            w2 = 1.0 - w1
            ensemble_oof = w1 * cat_oof_t + w2 * lgb_oof_t
            
            for t in np.arange(0.25, 0.65, 0.005):
                preds = (ensemble_oof >= t).astype(int)
                score = f1_score(y_full, preds)
                if score > best_ensemble_f1:
                    best_ensemble_f1 = score
                    best_weights = (round(w1, 2), round(w2, 2))
                    best_threshold = round(t, 3)
                    
        print(f"Optimal Ensemble Weights (CatBoost, LightGBM): {best_weights}")
        print(f"Optimal Decision Threshold: {best_threshold}")
        print(f"Winning CV F1 Score: {best_ensemble_f1:.5f}")
        
        # --- Submission Export ---
        final_proba = best_weights[0] * cat_test + best_weights[1] * lgb_test
        final_preds = (final_proba >= best_threshold).astype(int)
        
        sub_path = os.path.join(data_dir, 'submission.csv')
        sub = pd.DataFrame({
            'User_ID': private_test['User_ID'],
            'Converted': final_preds
        })
        sub.to_csv(sub_path, index=False)
        print(f"\nSaved final submission to '{sub_path}'. Shape: {sub.shape}")
        
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    run_pipeline()
