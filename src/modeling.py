from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class SplitSummary:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


def build_preprocessor(X: pd.DataFrame, categorical: List[str]) -> ColumnTransformer:
    numeric = [c for c in X.columns if c not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), categorical),
        ],
        remainder='drop',
    )
    return preprocessor


def model_performance(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        'AUC': roc_auc_score(y_true, y_proba),
        'PR_AUC': average_precision_score(y_true, y_proba),
        'Brier': brier_score_loss(y_true, y_proba),
        'F1': f1_score(y_true, (y_proba >= 0.5).astype(int)),
    }


def make_cv_folds(y: pd.Series, n_splits: int = 5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def random_split_baseline_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target: str = 'LOAN_DEFAULT'):
    """Create a small summary table comparing a random split to the temporal split."""
    df_all = pd.concat([train, val, test], ignore_index=True)
    train_random, temp = df_all.sample(frac=0.7, random_state=42).sort_index(), df_all.drop(df_all.sample(frac=0.7, random_state=42).index)
    return {'train_size': len(train), 'val_size': len(val), 'test_size': len(test)}
