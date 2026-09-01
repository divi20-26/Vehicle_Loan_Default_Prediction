# Project Vehi EMI - Loan Default Prediction Pipeline

## Project Overview
This project implements a production-ready machine learning pipeline for predicting vehicle EMI (Equated Monthly Installment) loan defaults using temporal validation and multiple optimization techniques.

## Pipeline Components

### 1. **Proper Optuna Tuning Loop**
- **XGBoost**: 5 Optuna trials for hyperparameter optimization
- **CatBoost**: 5 Optuna trials for hyperparameter optimization
- Baseline models: Logistic Regression, Random Forest, Extra Trees, Histogram Gradient Boosting
- Objective: Maximize AUC on validation set using Bayesian optimization (TPE sampler)

### 2. **SHAP Explanation Output**
- SHAP values computed on validation sample (250 instances)
- Feature importance ranking saved to CSV
- Lightweight approach (no heavy plotting) for production efficiency

### 3. **Confidence-Gated Triage Layer**
- Three-tier decision framework based on prediction confidence:
  - **High Confidence (≥0.80)**: Auto-approve (default=0) or escalate for review (default=1)
  - **Medium Confidence (0.60-0.80)**: Manual review required
  - **Low Confidence (<0.60)**: Manual review required
- Triage decisions saved to CSV for downstream processing

## Dataset
- **Source**: Vehicle EMI Loan Default Data (233,154 instances, 41 features)
- **Target**: LOAN_DEFAULT (binary classification)
- **Temporal Period**: 2018-08-01 to 2018-10-31
- **Default Rate**: 21.7%

## Temporal Validation Split
- **Training**: 2018-08-01 to 2018-10-16 (157,889 samples)
- **Validation**: 2018-10-16 to 2018-10-25 (38,254 samples)
- **Test (Out-of-time)**: 2018-10-25 to 2018-10-31 (37,011 samples)

## Verified Results (Out-of-Time AUC)
| Model | AUC | Accuracy | Brier | PR-AUC |
|-------|-----|----------|-------|--------|
| Logistic Regression | 0.6176 | 0.7354 | 0.1893 | 0.3467 |
| Random Forest | 0.6344 | 0.7360 | 0.1873 | 0.3649 |
| Extra Trees | 0.6246 | 0.7361 | 0.1885 | 0.3570 |
| Histogram Gradient Boosting | 0.6431 | 0.7359 | 0.1859 | 0.3758 |
| **XGBoost (Optuna-tuned)** | **0.6514** | **0.7368** | **0.1848** | **0.3862** |
| CatBoost (Optuna-tuned) | 0.6494 | 0.7361 | 0.1851 | 0.3831 |

**Best Model**: XGBoost with Optuna tuning (AUC: 0.6514)

## Leakage Features Removed
- UNIQUEID, PERFORM_CNS_SCORE_DESCRIPTION
- Primary account features (PRI_NO_OF_ACCTS, PRI_ACTIVE_ACCTS, etc.)
- Secondary account features (SEC_NO_OF_ACCTS, SEC_ACTIVE_ACCTS, etc.)
- Installment amounts (PRIMARY_INSTAL_AMT, SEC_INSTAL_AMT)
- Bureau derived features (NEW_ACCTS_IN_LAST_SIX_MONTHS, etc.)

## Key Insights
1. **Realistic Metrics**: Under strict temporal validation, the best model achieves ~65% AUC and ~74% accuracy
2. **Leakage Protection**: Systematic removal of future-information features ensures generalization
3. **Calibration**: All models use calibrated probability estimates via CalibratedClassifierCV
4. **Production Ready**: Complete pipeline with explainability and triage decision support

## Output Files
- `model_summary.csv`: Performance metrics for all 6 models
- `model_auc.png`: Visualization of model AUC comparison
- `triage_decisions.csv`: Confidence-gated decisions for each test prediction
- `*_shap_summary.csv`: Top SHAP features for best model
- `xgb_shap_beeswarm.png`: SHAP force plot visualization

## Usage
```bash
python src/run_pipeline.py
```

Expected runtime: ~15-20 minutes (Optuna tuning included)

## Dependencies
- pandas, numpy, scikit-learn
- xgboost, catboost
- optuna (Bayesian hyperparameter optimization)
- shap (SHAP values for explainability)
- matplotlib (visualization)
