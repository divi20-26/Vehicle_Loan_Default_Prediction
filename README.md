# Vehicle Loan Default Prediction

This project investigates vehicle-loan default prediction using a temporal split, leakage audit, feature engineering, calibrated models, SHAP explanations, and a confidence-gated triage layer.

## Dataset

- Source: Kaggle vehicle loan default prediction dataset
- Local files in `archive/`
  - `train.csv`
  - `test.csv`
  - `Data Dictionary.xlsx`

## Key methodology

1. Parse date fields carefully (`DisbursalDate` and `Date.of.Birth`), including century correction for 2-digit-year values where needed.
2. Sort by disbursal date and create a true out-of-time split:
   - Train: earliest ~70% of vintages
   - Validation: next ~15%
   - Test: most recent ~15%
3. Perform a leakage audit before modeling and remove post-first-EMI fields.
4. Build comparable models with matched tuning budgets.
5. Calibrate probabilities on validation.
6. Explain models with SHAP and assess cross-model stability.
7. Route applications through cost-aware triage.

## Project structure

- `src/` : reusable pipeline code
- `notebooks/` : optional exploratory notebook
- `archive/` : raw data files

## Run the pipeline

```bash
python src/run_pipeline.py
```

## Notes

This project is intended to be run from the repository root.
