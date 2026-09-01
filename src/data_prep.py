from __future__ import annotations

import re
from typing import List

import pandas as pd


def parse_date_series(series: pd.Series) -> pd.Series:
    """Parse the messy dates in the loan dataset with reliable day-first handling."""
    s = series.astype(str).str.strip()
    s = s.replace({'nan': '', 'NaN': '', 'None': '', 'NULL': ''})

    parsed = pd.to_datetime(s, dayfirst=True, errors='coerce')

    # Hard-case fix: some two-digit years can slip through without a century.
    for idx, value in s.items():
        if pd.isna(parsed.at[idx]):
            v = value.strip()
            if not v:
                continue
            m = re.match(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{2})$', v)
            if not m:
                continue
            d, mo, yy = m.groups()
            year = 2000 + int(yy) if int(yy) <= 20 else 1900 + int(yy)
            parsed.at[idx] = pd.Timestamp(year=year, month=int(mo), day=int(d))

    return parsed


def create_out_of_time_split(df: pd.DataFrame, date_col: str = 'DISBURSAL_DATE', train_frac: float = 0.70, val_frac: float = 0.15):
    """Create a time-ordered train/validation/test split."""
    if date_col not in df.columns:
        raise KeyError(f'{date_col} not found in dataframe')

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.sort_values(date_col).reset_index(drop=True)
    n = len(data)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = data.iloc[:train_end].copy()
    val = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()
    return train, val, test


def identify_leakage_features(df: pd.DataFrame) -> List[str]:
    """List known leakage-ish fields that should be dropped before modeling."""
    leakage = [
        'UNIQUEID',
        'PERFORM_CNS_SCORE_DESCRIPTION',
        'PRI_NO_OF_ACCTS',
        'PRI_ACTIVE_ACCTS',
        'PRI_OVERDUE_ACCTS',
        'PRI_CURRENT_BALANCE',
        'PRI_SANCTIONED_AMOUNT',
        'PRI_DISBURSED_AMOUNT',
        'SEC_NO_OF_ACCTS',
        'SEC_ACTIVE_ACCTS',
        'SEC_OVERDUE_ACCTS',
        'SEC_CURRENT_BALANCE',
        'SEC_SANCTIONED_AMOUNT',
        'SEC_DISBURSED_AMOUNT',
        'PRIMARY_INSTAL_AMT',
        'SEC_INSTAL_AMT',
        'NEW_ACCTS_IN_LAST_SIX_MONTHS',
        'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS',
        'AVERAGE_ACCT_AGE',
        'CREDIT_HISTORY_LENGTH',
        'NO_OF_INQUIRIES',
    ]
    return [c for c in leakage if c in df.columns]


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')
