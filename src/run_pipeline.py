from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap

from catboost import CatBoostClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

from data_prep import (
    create_out_of_time_split,
    identify_leakage_features,
    parse_date_series,
)
from feature_engineering import engineering_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_ROOT / 'archive'


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ARCHIVE_DIR / 'train.csv')
    return df


def safe_date_parse(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data['DISBURSAL_DATE'] = parse_date_series(
        data['DISBURSAL_DATE']
    )

    data['DATE_OF_BIRTH'] = parse_date_series(
        data['DATE_OF_BIRTH']
    )

    return data


def leakage_drop_columns(df: pd.DataFrame) -> List[str]:
    leak = identify_leakage_features(df)

    return list(
        dict.fromkeys(
            leak + ['PERFORM_CNS_SCORE_DESCRIPTION']
        )
    )


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = engineering_features(df)

    data['bureau_score_numeric'] = pd.to_numeric(
        data['PERFORM_CNS_SCORE'],
        errors='coerce'
    )

    data['ltv_band'] = pd.Categorical(
        data['ltv_band']
    )

    data['bureau_score_band'] = pd.Categorical(
        data['bureau_score_band']
    )

    data['loan_to_asset_ratio'] = data[
        'loan_to_asset_ratio'
    ].clip(
        lower=0,
        upper=5
    )

    data['sanction_minus_disbursal'] = data[
        'sanction_minus_disbursal'
    ].clip(
        lower=-500000,
        upper=500000
    )

    return data


def build_preprocessor(
    X: pd.DataFrame
) -> ColumnTransformer:

    # FIX:
    # Explicitly identify object, string and categorical columns.
    # This avoids the deprecated is_categorical_dtype().
    cat_cols = X.select_dtypes(
        include=['object', 'string', 'category']
    ).columns.tolist()

    num_cols = [
        c for c in X.columns
        if c not in cat_cols
    ]

    return ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline([
                    (
                        'imputer',
                        SimpleImputer(
                            strategy='median'
                        )
                    ),
                    (
                        'scaler',
                        StandardScaler()
                    ),
                ]),
                num_cols,
            ),
            (
                'cat',
                Pipeline([
                    (
                        'imputer',
                        SimpleImputer(
                            strategy='most_frequent'
                        )
                    ),
                    (
                        'onehot',
                        OneHotEncoder(
                            handle_unknown='ignore'
                        )
                    ),
                ]),
                cat_cols,
            ),
        ],
        remainder='drop',
    )


def normalize_categorical_columns(
    frame: pd.DataFrame,
    cat_cols: List[str]
) -> pd.DataFrame:

    out = frame.copy()

    for col in cat_cols:

        if col not in out.columns:
            continue

        out[col] = (
            out[col]
            .fillna('missing')
            .astype(str)
            .str.replace(
                'nan',
                'missing',
                regex=False
            )
            .str.replace(
                'None',
                'missing',
                regex=False
            )
            .str.replace(
                '<NA>',
                'missing',
                regex=False
            )
        )

    return out


def evaluate(
    y_true: pd.Series,
    proba: np.ndarray
) -> Dict[str, float]:

    pred = (
        proba >= 0.5
    ).astype(int)

    return {
        'AUC': float(
            roc_auc_score(
                y_true,
                proba
            )
        ),

        'PR_AUC': float(
            average_precision_score(
                y_true,
                proba
            )
        ),

        'Brier': float(
            brier_score_loss(
                y_true,
                proba
            )
        ),

        'Accuracy': float(
            accuracy_score(
                y_true,
                pred
            )
        ),
    }


def random_split_comparison(
    X: pd.DataFrame,
    y: pd.Series
) -> Dict[str, float]:

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    model = Pipeline([
        (
            'pre',
            build_preprocessor(X_train)
        ),

        (
            'model',
            LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                solver='lbfgs'
            )
        ),
    ])

    model.fit(
        X_train,
        y_train
    )

    random_proba = model.predict_proba(
        X_test
    )[:, 1]

    return evaluate(
        y_test,
        random_proba
    )


def optuna_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
):

    X_train_enc = pd.get_dummies(
        X_train,
        drop_first=False
    )

    X_val_enc = pd.get_dummies(
        X_val,
        drop_first=False
    )

    X_train_enc, X_val_enc = X_train_enc.align(
        X_val_enc,
        join='outer',
        axis=1,
        fill_value=0
    )

    def objective(trial):

        params = {
            'n_estimators': trial.suggest_int(
                'n_estimators',
                400,
                1200
            ),

            'max_depth': trial.suggest_int(
                'max_depth',
                3,
                10
            ),

            'learning_rate': trial.suggest_float(
                'learning_rate',
                0.01,
                0.1,
                log=True
            ),

            'subsample': trial.suggest_float(
                'subsample',
                0.6,
                1.0
            ),

            'colsample_bytree': trial.suggest_float(
                'colsample_bytree',
                0.6,
                1.0
            ),

            'min_child_weight': trial.suggest_int(
                'min_child_weight',
                1,
                8
            ),

            'reg_lambda': trial.suggest_float(
                'reg_lambda',
                1e-3,
                5.0,
                log=True
            ),

            'scale_pos_weight': trial.suggest_float(
                'scale_pos_weight',
                1.0,
                8.0
            ),
        }

        model = XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1,
            **params
        )

        model.fit(
            X_train_enc,
            y_train
        )

        proba = model.predict_proba(
            X_val_enc
        )[:, 1]

        return roc_auc_score(
            y_val,
            proba
        )

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            seed=42
        )
    )

    study.optimize(
        objective,
        n_trials=30
    )

    best = study.best_params

    return XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
        **best
    )


def optuna_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
):

    # FIX:
    # Explicit categorical-column detection.
    cat_cols = X_train.select_dtypes(
        include=['object', 'string', 'category']
    ).columns.tolist()

    X_train_cb = normalize_categorical_columns(
        X_train,
        cat_cols
    )

    X_val_cb = normalize_categorical_columns(
        X_val,
        cat_cols
    )

    def objective(trial):

        params = {
            'iterations': trial.suggest_int(
                'iterations',
                400,
                1200
            ),

            'depth': trial.suggest_int(
                'depth',
                4,
                10
            ),

            'learning_rate': trial.suggest_float(
                'learning_rate',
                0.01,
                0.1,
                log=True
            ),

            'l2_leaf_reg': trial.suggest_float(
                'l2_leaf_reg',
                1e-3,
                10.0,
                log=True
            ),

            'bagging_temperature': trial.suggest_float(
                'bagging_temperature',
                0.0,
                3.0
            ),

            'auto_class_weights': 'Balanced',

            'loss_function': 'Logloss',

            'eval_metric': 'AUC',

            'random_seed': 42,

            'verbose': False,

            'custom_metric': ['AUC'],
        }

        model = CatBoostClassifier(
            cat_features=cat_cols,
            **params
        )

        model.fit(
            X_train_cb,
            y_train,
            eval_set=(X_val_cb, y_val),
            verbose=False
        )

        proba = model.predict_proba(
            X_val_cb
        )[:, 1]

        return roc_auc_score(
            y_val,
            proba
        )

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            seed=42
        )
    )

    study.optimize(
        objective,
        n_trials=30
    )

    best = study.best_params

    return CatBoostClassifier(
        cat_features=cat_cols,
        **best,
        auto_class_weights='Balanced',
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=False
    )


def build_triage_table(
    proba: np.ndarray,
    y_true: pd.Series
) -> pd.DataFrame:

    df = pd.DataFrame({
        'probability': proba,

        'predicted_label': (
            proba >= 0.5
        ).astype(int),

        'actual_label': y_true.astype(int),

        'confidence': np.maximum(
            proba,
            1.0 - proba
        ),
    })

    df['decision'] = 'manual_review'

    df.loc[
        (df['confidence'] >= 0.80)
        & (df['predicted_label'] == 1),
        'decision'
    ] = 'escalate_for_review'

    df.loc[
        (df['confidence'] >= 0.80)
        & (df['predicted_label'] == 0),
        'decision'
    ] = 'auto_approve'

    df.loc[
        df['confidence'] < 0.60,
        'decision'
    ] = 'manual_review'

    return df


def save_shap_summary(
    model,
    X_eval: pd.DataFrame,
    output_dir: Path,
    name: str
) -> pd.DataFrame:

    sample_size = min(
        250,
        len(X_eval)
    )

    sample = X_eval.sample(
        n=sample_size,
        random_state=42
    )

    if hasattr(model, 'named_steps'):

        pre = model.named_steps.get(
            'pre'
        )

        estimator = model.named_steps.get(
            'model'
        )

        if pre is not None:

            transformed = pre.transform(
                sample
            )

            feature_names = (
                pre
                .get_feature_names_out()
                .tolist()
            )

            explainer = shap.Explainer(
                estimator,
                transformed
            )

            vals = explainer(
                transformed
            )

            values = np.asarray(
                vals.values
            )

            if values.ndim == 3:
                values = values[:, :, 1]

            if (
                values.ndim == 2
                and values.shape[1]
                != len(feature_names)
            ):
                values = values[
                    :,
                    :len(feature_names)
                ]

            summary = pd.DataFrame({
                'feature': feature_names,

                'mean_abs_shap': (
                    np.abs(values)
                    .mean(axis=0)
                ),
            }).sort_values(
                'mean_abs_shap',
                ascending=False
            ).reset_index(
                drop=True
            )

            summary.to_csv(
                output_dir
                / f'{name}_shap_summary.csv',
                index=False
            )

            return summary

    if hasattr(
        model,
        'feature_names_in_'
    ):

        feature_names = list(
            model.feature_names_in_
        )

        explainer = shap.Explainer(
            model
        )

        vals = explainer(
            sample
        )

        values = np.asarray(
            vals.values
        )

        if values.ndim == 3:
            values = values[:, :, 1]

        summary = pd.DataFrame({
            'feature': feature_names,

            'mean_abs_shap': (
                np.abs(values)
                .mean(axis=0)
            ),
        }).sort_values(
            'mean_abs_shap',
            ascending=False
        ).reset_index(
            drop=True
        )

        summary.to_csv(
            output_dir
            / f'{name}_shap_summary.csv',
            index=False
        )

        return summary

    return pd.DataFrame()


def fit_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, object]:

    # ---------------------------------------------------------
    # LOGISTIC REGRESSION
    # ---------------------------------------------------------

    if model_name == 'logit':

        pipe = Pipeline([
            (
                'pre',
                build_preprocessor(X_train)
            ),

            (
                'model',
                LogisticRegression(
                    max_iter=2000,
                    class_weight='balanced',
                    solver='lbfgs'
                )
            ),
        ])

        pipe.fit(
            X_train,
            y_train
        )

        # FIX:
        # FrozenEstimator replaces deprecated cv='prefit'.
        cal = CalibratedClassifierCV(
            FrozenEstimator(pipe),
            method='sigmoid'
        )

        cal.fit(
            X_val,
            y_val
        )

        proba = cal.predict_proba(
            X_test
        )[:, 1]

        return {
            'model': cal,

            'scores': evaluate(
                y_test,
                proba
            ),

            'val_scores': evaluate(
                y_val,
                cal.predict_proba(
                    X_val
                )[:, 1]
            ),

            'val_proba': cal.predict_proba(
                X_val
            )[:, 1],

            'base_model': pipe,

            'proba': proba
        }

    # ---------------------------------------------------------
    # RANDOM FOREST
    # ---------------------------------------------------------

    if model_name == 'rf':

        pipe = Pipeline([
            (
                'pre',
                build_preprocessor(X_train)
            ),

            (
                'model',
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=18,
                    min_samples_leaf=4,
                    class_weight='balanced_subsample',
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        ])

        pipe.fit(
            X_train,
            y_train
        )

        # FIX:
        # FrozenEstimator replaces deprecated cv='prefit'.
        cal = CalibratedClassifierCV(
            FrozenEstimator(pipe),
            method='sigmoid'
        )

        cal.fit(
            X_val,
            y_val
        )

        proba = cal.predict_proba(
            X_test
        )[:, 1]

        return {
            'model': cal,

            'scores': evaluate(
                y_test,
                proba
            ),

            'val_scores': evaluate(
                y_val,
                cal.predict_proba(
                    X_val
                )[:, 1]
            ),

            'val_proba': cal.predict_proba(
                X_val
            )[:, 1],

            'base_model': pipe,

            'proba': proba
        }

    # ---------------------------------------------------------
    # EXTRA TREES
    # ---------------------------------------------------------

    if model_name == 'extra_trees':

        pipe = Pipeline([
            (
                'pre',
                build_preprocessor(X_train)
            ),

            (
                'model',
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        ])

        pipe.fit(
            X_train,
            y_train
        )

        # FIX:
        # FrozenEstimator replaces deprecated cv='prefit'.
        cal = CalibratedClassifierCV(
            FrozenEstimator(pipe),
            method='sigmoid'
        )

        cal.fit(
            X_val,
            y_val
        )

        proba = cal.predict_proba(
            X_test
        )[:, 1]

        return {
            'model': cal,

            'scores': evaluate(
                y_test,
                proba
            ),

            'val_scores': evaluate(
                y_val,
                cal.predict_proba(
                    X_val
                )[:, 1]
            ),

            'val_proba': cal.predict_proba(
                X_val
            )[:, 1],

            'base_model': pipe,

            'proba': proba
        }

    # ---------------------------------------------------------
    # HISTOGRAM GRADIENT BOOSTING
    # ---------------------------------------------------------

    if model_name == 'hgb':

        pipe = Pipeline([
            (
                'pre',
                build_preprocessor(X_train)
            ),

            (
                'model',
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=8,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    random_state=42,
                )
            ),
        ])

        pipe.fit(
            X_train,
            y_train
        )

        # FIX:
        # FrozenEstimator replaces deprecated cv='prefit'.
        cal = CalibratedClassifierCV(
            FrozenEstimator(pipe),
            method='sigmoid'
        )

        cal.fit(
            X_val,
            y_val
        )

        proba = cal.predict_proba(
            X_test
        )[:, 1]

        return {
            'model': cal,

            'scores': evaluate(
                y_test,
                proba
            ),

            'val_scores': evaluate(
                y_val,
                cal.predict_proba(
                    X_val
                )[:, 1]
            ),

            'val_proba': cal.predict_proba(
                X_val
            )[:, 1],

            'base_model': pipe,

            'proba': proba
        }

    # ---------------------------------------------------------
    # XGBOOST
    # ---------------------------------------------------------

    if model_name == 'xgb':

        X_train_enc = pd.get_dummies(
            X_train,
            drop_first=False
        )

        X_val_enc = pd.get_dummies(
            X_val,
            drop_first=False
        )

        X_test_enc = pd.get_dummies(
            X_test,
            drop_first=False
        )

        X_train_enc, X_val_enc = (
            X_train_enc.align(
                X_val_enc,
                join='outer',
                axis=1,
                fill_value=0
            )
        )

        X_train_enc, X_test_enc = (
            X_train_enc.align(
                X_test_enc,
                join='outer',
                axis=1,
                fill_value=0
            )
        )

        tuned = optuna_xgb(
            X_train,
            y_train,
            X_val,
            y_val
        )

        tuned.fit(
            X_train_enc,
            y_train
        )

        # FIX:
        # FrozenEstimator replaces deprecated cv='prefit'.
        cal = CalibratedClassifierCV(
            FrozenEstimator(tuned),
            method='sigmoid'
        )

        cal.fit(
            X_val_enc,
            y_val
        )

        proba = cal.predict_proba(
            X_test_enc
        )[:, 1]

        return {
            'model': cal,

            'scores': evaluate(
                y_test,
                proba
            ),

            'val_scores': evaluate(
                y_val,
                cal.predict_proba(
                    X_val_enc
                )[:, 1]
            ),

            'val_proba': cal.predict_proba(
                X_val_enc
            )[:, 1],

            'base_model': tuned,

            'feature_matrix': X_val_enc,

            'proba': proba
        }

    # ---------------------------------------------------------
    # CATBOOST
    # ---------------------------------------------------------

    if model_name == 'catboost':

        # FIX:
        # Explicit categorical-column detection.
        cat_cols = X_train.select_dtypes(
            include=[
                'object',
                'string',
                'category'
            ]
        ).columns.tolist()

        X_train_cb = (
            normalize_categorical_columns(
                X_train,
                cat_cols
            )
        )

        X_val_cb = (
            normalize_categorical_columns(
                X_val,
                cat_cols
            )
        )

        X_test_cb = (
            normalize_categorical_columns(
                X_test,
                cat_cols
            )
        )

        tuned = optuna_catboost(
            X_train_cb,
            y_train,
            X_val_cb,
            y_val
        )

        tuned.fit(
            X_train_cb,
            y_train,
            eval_set=(
                X_val_cb,
                y_val
            ),
            verbose=False
        )

        # FIX:
        # FrozenEstimator replaces deprecated cv='prefit'.
        cal = CalibratedClassifierCV(
            FrozenEstimator(tuned),
            method='sigmoid'
        )

        cal.fit(
            X_val_cb,
            y_val
        )

        proba = cal.predict_proba(
            X_test_cb
        )[:, 1]

        return {
            'model': cal,

            'scores': evaluate(
                y_test,
                proba
            ),

            'val_scores': evaluate(
                y_val,
                cal.predict_proba(
                    X_val_cb
                )[:, 1]
            ),

            'val_proba': cal.predict_proba(
                X_val_cb
            )[:, 1],

            'base_model': tuned,

            'feature_matrix': X_val_cb,

            'proba': proba
        }

    raise ValueError(
        f'Unsupported model: {model_name}'
    )


def summarize_split_dates(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> Dict[str, str]:

    return {
        'train_start': str(
            train[
                'DISBURSAL_DATE'
            ].min().date()
        ),

        'train_end': str(
            train[
                'DISBURSAL_DATE'
            ].max().date()
        ),

        'val_start': str(
            val[
                'DISBURSAL_DATE'
            ].min().date()
        ),

        'val_end': str(
            val[
                'DISBURSAL_DATE'
            ].max().date()
        ),

        'test_start': str(
            test[
                'DISBURSAL_DATE'
            ].min().date()
        ),

        'test_end': str(
            test[
                'DISBURSAL_DATE'
            ].max().date()
        ),
    }


def build_ensemble_analysis(
    model_probas: Dict[str, np.ndarray],
    y_test: pd.Series,
    output_dir: Path
) -> Dict[str, object]:

    # Create probability matrix
    pred_df = pd.DataFrame(model_probas)

    # Equal-weight ensemble
    pred_df['ensemble_probability'] = (
        pred_df.mean(axis=1)
    )

    # Model disagreement / uncertainty
    pred_df['prediction_variance'] = (
        pred_df.var(axis=1, ddof=0)
    )

    pred_df['prediction_std'] = (
        pred_df.std(axis=1, ddof=0)
    )

    # Final ensemble prediction
    pred_df['ensemble_prediction'] = (
        pred_df['ensemble_probability'] >= 0.5
    ).astype(int)

    pred_df['actual_label'] = (
        y_test.to_numpy()
    )

    # Correct / incorrect
    pred_df['correct'] = (
        pred_df['ensemble_prediction']
        == pred_df['actual_label']
    ).astype(int)

    # Save complete prediction matrix
    pred_df.to_csv(
        output_dir / 'model_predictions.csv',
        index=False
    )

    # Evaluate ensemble
    ensemble_scores = evaluate(
        y_test,
        pred_df['ensemble_probability'].to_numpy()
    )

    # ---------------------------------------------------------
    # UNCERTAINTY ANALYSIS
    # ---------------------------------------------------------

    # Divide applicants into variance quartiles
    pred_df['uncertainty_group'] = pd.qcut(
        pred_df['prediction_variance'],
        q=4,
        labels=[
            'Q1_low',
            'Q2',
            'Q3',
            'Q4_high'
        ],
        duplicates='drop'
    )

    uncertainty_summary = (
        pred_df
        .groupby(
            'uncertainty_group',
            observed=True
        )
        .agg(
            samples=('correct', 'size'),
            error_rate=(
                'correct',
                lambda x: 1 - x.mean()
            ),
            mean_variance=(
                'prediction_variance',
                'mean'
            ),
            mean_probability=(
                'ensemble_probability',
                'mean'
            )
        )
        .reset_index()
    )

    uncertainty_summary.to_csv(
        output_dir / 'uncertainty_analysis.csv',
        index=False
    )

    print(
        'ENSEMBLE_RESULTS',
        json.dumps(
            ensemble_scores,
            sort_keys=True
        )
    )

    print(
        'UNCERTAINTY_ANALYSIS'
    )

    print(
        uncertainty_summary.to_string(
            index=False
        )
    )

    return {
        'predictions': pred_df,
        'scores': ensemble_scores,
        'uncertainty_summary': uncertainty_summary
    }



def _routing_cost(
    decisions: np.ndarray,
    y_true: np.ndarray
) -> np.ndarray:

    # Normalized research costs. These are not claimed to be
    # real lender costs.
    false_safe_cost = 5.0
    manual_review_cost = 1.0
    false_escalate_cost = 2.0

    costs = np.zeros(
        len(y_true),
        dtype=float
    )

    auto_mask = decisions == 'auto_process'
    manual_mask = decisions == 'manual_review'
    escalate_mask = decisions == 'escalate_for_review'

    costs[
        auto_mask & (y_true == 1)
    ] = false_safe_cost

    costs[manual_mask] = manual_review_cost

    costs[
        escalate_mask & (y_true == 0)
    ] = false_escalate_cost

    return costs


def _make_routing_decisions(
    probability: np.ndarray,
    variance: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    variance_threshold: float | None = None
) -> np.ndarray:

    decisions = np.full(
        len(probability),
        'manual_review',
        dtype=object
    )

    if variance_threshold is None:
        confident = np.ones(
            len(probability),
            dtype=bool
        )
    else:
        confident = variance < variance_threshold

    decisions[
        confident & (probability < low_threshold)
    ] = 'auto_process'

    decisions[
        confident & (probability >= high_threshold)
    ] = 'escalate_for_review'

    return decisions


def build_cost_aware_routing(
    model_val_probas: Dict[str, np.ndarray],
    ensemble_result: Dict[str, object],
    y_val: pd.Series,
    output_dir: Path
) -> Dict[str, object]:

    # ---------------------------------------------------------
    # VALIDATION PROBABILITIES
    # ---------------------------------------------------------

    val_df = pd.DataFrame(
        model_val_probas
    )

    val_probability = val_df.mean(
        axis=1
    ).to_numpy()

    val_variance = val_df.var(
        axis=1,
        ddof=0
    ).to_numpy()

    y_val_np = y_val.to_numpy()

    # ---------------------------------------------------------
    # POLICY SEARCH
    # ---------------------------------------------------------
    #
    # The validation set selects the probability and uncertainty
    # thresholds. The temporal test set is used only afterwards
    # for final evaluation.
    # ---------------------------------------------------------

    probability_grid = np.linspace(
        0.10,
        0.45,
        15
    )

    high_grid = np.linspace(
        0.55,
        0.90,
        15
    )

    variance_grid = np.quantile(
        val_variance,
        np.linspace(0.50, 0.95, 10)
    )

    best_probability_cost = float('inf')
    best_probability_policy = None

    best_uncertainty_cost = float('inf')
    best_uncertainty_policy = None

    # Probability-only policy
    for low_t in probability_grid:

        for high_t in high_grid:

            if low_t >= high_t:
                continue

            decisions = _make_routing_decisions(
                val_probability,
                val_variance,
                low_t,
                high_t
            )

            costs = _routing_cost(
                decisions,
                y_val_np
            )

            mean_cost = costs.mean()

            if mean_cost < best_probability_cost:
                best_probability_cost = mean_cost
                best_probability_policy = {
                    'low_probability_threshold': float(low_t),
                    'high_probability_threshold': float(high_t),
                    'validation_mean_cost': float(mean_cost)
                }

    # Probability + uncertainty policy
    for low_t in probability_grid:

        for high_t in high_grid:

            if low_t >= high_t:
                continue

            for var_t in variance_grid:

                decisions = _make_routing_decisions(
                    val_probability,
                    val_variance,
                    low_t,
                    high_t,
                    float(var_t)
                )

                costs = _routing_cost(
                    decisions,
                    y_val_np
                )

                mean_cost = costs.mean()

                if mean_cost < best_uncertainty_cost:
                    best_uncertainty_cost = mean_cost
                    best_uncertainty_policy = {
                        'low_probability_threshold': float(low_t),
                        'high_probability_threshold': float(high_t),
                        'variance_threshold': float(var_t),
                        'validation_mean_cost': float(mean_cost)
                    }

    if best_probability_policy is None:
        raise RuntimeError(
            'Could not determine probability-only routing policy.'
        )

    if best_uncertainty_policy is None:
        raise RuntimeError(
            'Could not determine uncertainty-aware routing policy.'
        )

    # ---------------------------------------------------------
    # TEST DATA FROM PREVIOUS ENSEMBLE ANALYSIS
    # ---------------------------------------------------------

    pred_df = ensemble_result[
        'predictions'
    ].copy()

    test_probability = pred_df[
        'ensemble_probability'
    ].to_numpy()

    test_variance = pred_df[
        'prediction_variance'
    ].to_numpy()

    y_test = pred_df[
        'actual_label'
    ].to_numpy()

    # ---------------------------------------------------------
    # BASELINE: ALWAYS MANUAL REVIEW
    # ---------------------------------------------------------

    always_manual = np.full(
        len(y_test),
        'manual_review',
        dtype=object
    )

    always_manual_cost = _routing_cost(
        always_manual,
        y_test
    ).mean()

    # ---------------------------------------------------------
    # TEST: PROBABILITY-ONLY POLICY
    # ---------------------------------------------------------

    prob_policy = best_probability_policy

    prob_only_decisions = _make_routing_decisions(
        test_probability,
        test_variance,
        prob_policy['low_probability_threshold'],
        prob_policy['high_probability_threshold']
    )

    prob_only_costs = _routing_cost(
        prob_only_decisions,
        y_test
    )

    # ---------------------------------------------------------
    # TEST: UNCERTAINTY-AWARE POLICY
    # ---------------------------------------------------------

    uncertainty_policy = best_uncertainty_policy

    uncertainty_decisions = _make_routing_decisions(
        test_probability,
        test_variance,
        uncertainty_policy['low_probability_threshold'],
        uncertainty_policy['high_probability_threshold'],
        uncertainty_policy['variance_threshold']
    )

    uncertainty_costs = _routing_cost(
        uncertainty_decisions,
        y_test
    )

    pred_df[
        'probability_only_decision'
    ] = prob_only_decisions

    pred_df[
        'uncertainty_aware_decision'
    ] = uncertainty_decisions

    pred_df[
        'probability_only_cost'
    ] = prob_only_costs

    pred_df[
        'uncertainty_aware_cost'
    ] = uncertainty_costs

    pred_df.to_csv(
        output_dir / 'cost_aware_routing.csv',
        index=False
    )

    # ---------------------------------------------------------
    # ROUTING SUMMARIES
    # ---------------------------------------------------------

    def make_summary(
        decisions: np.ndarray,
        costs: np.ndarray,
        name: str
    ) -> pd.DataFrame:

        summary = (
            pd.Series(decisions)
            .value_counts()
            .rename_axis('decision')
            .reset_index(name='samples')
        )

        summary['percentage'] = (
            summary['samples']
            / len(decisions)
            * 100
        )

        summary['mean_cost'] = float(
            costs.mean()
        )

        summary['policy'] = name

        return summary

    probability_summary = make_summary(
        prob_only_decisions,
        prob_only_costs,
        'probability_only'
    )

    uncertainty_summary = make_summary(
        uncertainty_decisions,
        uncertainty_costs,
        'uncertainty_aware'
    )

    routing_summary = pd.concat(
        [
            probability_summary,
            uncertainty_summary
        ],
        ignore_index=True
    )

    routing_summary.to_csv(
        output_dir / 'cost_aware_routing_summary.csv',
        index=False
    )

    # ---------------------------------------------------------
    # POLICY COMPARISON
    # ---------------------------------------------------------

    comparison = pd.DataFrame([
        {
            'policy': 'always_manual_review',
            'test_mean_cost': float(always_manual_cost)
        },
        {
            'policy': 'probability_only',
            'test_mean_cost': float(prob_only_costs.mean())
        },
        {
            'policy': 'uncertainty_aware',
            'test_mean_cost': float(uncertainty_costs.mean())
        }
    ])

    comparison.to_csv(
        output_dir / 'routing_cost_comparison.csv',
        index=False
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print(
        'COST_AWARE_POLICY',
        json.dumps(
            uncertainty_policy,
            sort_keys=True
        )
    )

    print(
        'PROBABILITY_ONLY_POLICY',
        json.dumps(
            prob_policy,
            sort_keys=True
        )
    )

    print(
        'ALWAYS_MANUAL_TEST_MEAN_COST',
        float(always_manual_cost)
    )

    print(
        'PROBABILITY_ONLY_TEST_MEAN_COST',
        float(prob_only_costs.mean())
    )

    print(
        'COST_AWARE_TEST_MEAN_COST',
        float(uncertainty_costs.mean())
    )

    print(
        'COST_AWARE_ROUTING'
    )

    print(
        uncertainty_summary.to_string(
            index=False
        )
    )

    return {
        'uncertainty_policy': uncertainty_policy,
        'probability_policy': prob_policy,
        'always_manual_cost': float(always_manual_cost),
        'probability_only_cost': float(prob_only_costs.mean()),
        'uncertainty_aware_cost': float(uncertainty_costs.mean()),
        'routing_summary': routing_summary,
        'predictions': pred_df
    }


def main() -> None:

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    df = load_data()

    df = safe_date_parse(
        df
    )

    df = df.sort_values(
        'DISBURSAL_DATE'
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # TEMPORAL SPLIT
    # ---------------------------------------------------------

    train, val, test = create_out_of_time_split(
        df,
        'DISBURSAL_DATE'
    )

    print(
        'OUT_OF_TIME_SPLIT_DATES'
    )

    print(
        json.dumps(
            summarize_split_dates(
                train,
                val,
                test
            ),
            indent=2
        )
    )

    # ---------------------------------------------------------
    # LEAKAGE COLUMNS
    # ---------------------------------------------------------

    leakage_cols = leakage_drop_columns(
        df
    )

    print(
        'LEAKAGE_COLUMNS_DROPPED',
        leakage_cols
    )

    # ---------------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------------

    feature_df = build_feature_frame(
        df
    )

    feature_df = feature_df.drop(
        columns=[
            'DISBURSAL_DATE',
            'DATE_OF_BIRTH'
        ],
        errors='ignore'
    )

    feature_df = feature_df.drop(
        columns=[
            c for c in leakage_cols
            if c in feature_df.columns
        ],
        errors='ignore'
    )

    feature_cols = [
        c for c in feature_df.columns
        if c not in [
            'LOAN_DEFAULT'
        ]
    ]

    # ---------------------------------------------------------
    # CREATE TRAIN / VALIDATION / TEST FEATURES
    # ---------------------------------------------------------

    train_feat = feature_df.iloc[
        :len(train)
    ].copy()

    val_feat = feature_df.iloc[
        len(train):
        len(train) + len(val)
    ].copy()

    test_feat = feature_df.iloc[
        len(train) + len(val):
    ].copy()

    y_train = train_feat[
        'LOAN_DEFAULT'
    ].astype(int)

    X_train = train_feat[
        feature_cols
    ]

    y_val = val_feat[
        'LOAN_DEFAULT'
    ].astype(int)

    X_val = val_feat[
        feature_cols
    ]

    y_test = test_feat[
        'LOAN_DEFAULT'
    ].astype(int)

    X_test = test_feat[
        feature_cols
    ]

    # ---------------------------------------------------------
    # RANDOM SPLIT COMPARISON
    # ---------------------------------------------------------

    random_stats = random_split_comparison(
        X_train,
        y_train
    )

    print(
        'RANDOM_SPLIT_VS_TEMPORAL_LOGIT',
        json.dumps(
            {
                'naive_random_split_auc':
                    random_stats['AUC']
            },
            indent=2
        )
    )

    # ---------------------------------------------------------
    # OUTPUT DIRECTORY
    # ---------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / 'outputs'
    )

    output_dir.mkdir(
        exist_ok=True
    )

    # ---------------------------------------------------------
    # MODEL TRAINING
    # ---------------------------------------------------------

    results = {}
    model_probas = {}
    model_val_probas = {}

    best_name = None

    best_score = -1.0

    best_proba = None

    best_result = None

    # Train all six baseline models
    for name in [
        'logit',
        'xgb',
        'rf',
        'extra_trees',
        'catboost',
        'hgb'
    ]:

        res = fit_model(
            name,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        )

        results[name] = res[
            'scores'
        ]
        model_probas[name] = res[
            'proba'
        ]

        model_val_probas[name] = res[
            'val_proba'
        ]

        print(
             'MODEL=' + name + ': '
             + json.dumps(
                  res['scores'],
                  sort_keys=True
            )
        )

        if (
            res['scores']['AUC']
            > best_score
        ):

            best_score = (
                res['scores']['AUC']
            )

            best_name = name

            best_result = res

            best_proba = res[
                'proba'
            ]
    # ---------------------------------------------------------
    # ENSEMBLE + UNCERTAINTY ANALYSIS
    # ---------------------------------------------------------

    ensemble_result = build_ensemble_analysis(
        model_probas,
        y_test,
        output_dir
    )

    # ---------------------------------------------------------
    # COST-AWARE UNCERTAINTY ROUTING
    # ---------------------------------------------------------

    routing_result = build_cost_aware_routing(
        model_val_probas,
        ensemble_result,
        y_val,
        output_dir
    )

    # ---------------------------------------------------------
    # SAVE OUTPUTS
    # ---------------------------------------------------------

    pd.DataFrame.from_dict(
        results,
        orient='index'
    ).to_csv(
        output_dir
        / 'model_summary.csv'
    )

    # ---------------------------------------------------------
    # AUC PLOT
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        list(results.keys()),
        [
            results[m]['AUC']
            for m in results
        ]
    )

    ax.set_title(
        'Out-of-time AUC by model'
    )

    ax.set_ylabel(
        'AUC'
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / 'model_auc.png',
        dpi=200
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # SHAP + TRIAGE
    # ---------------------------------------------------------

    if (
        best_result is not None
        and best_name is not None
    ):

        if 'feature_matrix' in best_result:

            shap_df = save_shap_summary(
                best_result[
                    'base_model'
                ],

                best_result[
                    'feature_matrix'
                ],

                output_dir,

                best_name
            )

        else:

            shap_df = save_shap_summary(
                best_result[
                    'base_model'
                ],

                X_val,

                output_dir,

                best_name
            )

        triage_df = build_triage_table(
            best_proba,
            y_test
        )

        triage_counts = (
            triage_df[
                'decision'
            ]
            .value_counts()
            .to_dict()
        )

        triage_df.to_csv(
            output_dir
            / 'triage_decisions.csv',
            index=False
        )

        print(
            'BEST_MODEL',
            best_name
        )

        print(
            'BEST_MODEL_AUC',
            best_score
        )

        print(
            'TRIAGE_SUMMARY',
            json.dumps(
                triage_counts,
                sort_keys=True
            )
        )

        if not shap_df.empty:

            print(
                'SHAP_TOP_FEATURES'
            )

            print(
                shap_df
                .head(10)
                .to_dict(
                    orient='records'
                )
            )

    print(
        'RESULTS_SAVED',
        str(output_dir)
    )

if __name__ == '__main__':
    main()