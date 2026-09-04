from __future__ import annotations

import numpy as np
import pandas as pd


def bucket_bureau_score(score: pd.Series) -> pd.Series:
    """Bucket the raw bureau score into risk bands, treating sentinel values as no-history."""
    s = pd.to_numeric(score, errors='coerce')
    bands = pd.Series('No Bureau History', index=s.index, dtype='object')
    bands[s.isna()] = 'No Bureau History'
    bands[s == 0] = 'No Bureau History'
    bands[(s > 0) & (s <= 300)] = 'High'
    bands[(s > 300) & (s <= 600)] = 'Medium'
    bands[(s > 600) & (s <= 900)] = 'Low'
    bands[(s > 900)] = 'Low'
    return bands


def engineering_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['DISBURSAL_DATE'] = pd.to_datetime(data['DISBURSAL_DATE'], errors='coerce')
    data['DATE_OF_BIRTH'] = pd.to_datetime(data['DATE_OF_BIRTH'], errors='coerce')

    # Missingness flag must be created before any imputation.
    data['bureau_score_missing'] = pd.to_numeric(data['PERFORM_CNS_SCORE'], errors='coerce').isna().astype(int)
    data['bureau_score_band'] = bucket_bureau_score(data['PERFORM_CNS_SCORE'])

    # Loan structure
    data['loan_to_asset_ratio'] = data['DISBURSED_AMOUNT'] / data['ASSET_COST']
    data['ltv_band'] = pd.cut(
        data['LTV'],
        bins=[0, 50, 75, 90, 100, 999],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        right=False,
    )
    total_sanctioned = data.get('PRI_SANCTIONED_AMOUNT', 0).fillna(0) + data.get('SEC_SANCTIONED_AMOUNT', 0).fillna(0)
    data['sanction_minus_disbursal'] = total_sanctioned - data['DISBURSED_AMOUNT']
    data['disbursal_gap_ratio'] = np.where(
        data['DISBURSED_AMOUNT'].replace(0, np.nan).notna(),
        data['sanction_minus_disbursal'] / data['DISBURSED_AMOUNT'],
        0.0,
    )

    # Demographics: compute age at origination from DOB + disbursal date.
    data['age_at_disbursal'] = ((data['DISBURSAL_DATE'] - data['DATE_OF_BIRTH']).dt.days / 365.25).clip(lower=0, upper=90)

    # Bureau aggregates
    total_accounts = data.get('PRI_NO_OF_ACCTS', 0).fillna(0) + data.get('SEC_NO_OF_ACCTS', 0).fillna(0)
    active_accounts = data.get('PRI_ACTIVE_ACCTS', 0).fillna(0) + data.get('SEC_ACTIVE_ACCTS', 0).fillna(0)
    data['active_to_total_bureau_ratio'] = (active_accounts / (total_accounts + 1e-6)).replace([np.inf, -np.inf], 0.0)
    data['closed_accounts'] = np.maximum(total_accounts - active_accounts, 0)
    data['total_overdue_accounts'] = data.get('PRI_OVERDUE_ACCTS', 0).fillna(0) + data.get('SEC_OVERDUE_ACCTS', 0).fillna(0)
    data['total_bureau_enquiries'] = data['NO_OF_INQUIRIES'].fillna(0)

    return data
