"""Exploratory data analysis for transaction fraud data.

Provides analysis functions specific to fraud detection:
- Class imbalance assessment
- Temporal fraud pattern analysis
- Feature distribution comparison (fraud vs. legitimate)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def class_balance_report(y: pd.Series) -> pd.DataFrame:
    """Report the class balance and imbalance ratio.

    Parameters
    ----------
    y : pd.Series
        Binary target (0=legit, 1=fraud).

    Returns
    -------
    pd.DataFrame
        Class counts, percentages, and imbalance ratio.
    """
    counts = y.value_counts().sort_index()
    records = []
    for cls in [0, 1]:
        label = "Legitimate" if cls == 0 else "Fraud"
        count = int(counts.get(cls, 0))
        records.append({
            "class": cls,
            "label": label,
            "count": count,
            "percentage": round(count / len(y) * 100, 3),
        })
    records.append({
        "class": -1,
        "label": "Imbalance Ratio",
        "count": int(counts.get(0, 1)),
        "percentage": round(counts.get(0, 1) / max(counts.get(1, 1), 1), 1),
    })
    return pd.DataFrame(records)


def fraud_rate_by_feature(
    X: pd.DataFrame,
    y: pd.Series,
    feature: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute the fraud rate within quantile bins of a feature.

    Parameters
    ----------
    X : pd.DataFrame
        Transaction features.
    y : pd.Series
        Fraud labels.
    feature : str
        Feature to analyse.
    n_bins : int
        Number of bins.

    Returns
    -------
    pd.DataFrame
        Binned fraud rates.
    """
    try:
        bins = pd.qcut(X[feature], q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(X[feature], bins=min(n_bins, X[feature].nunique()))

    grouped = pd.DataFrame({"bin": bins, "fraud": y}).groupby("bin", observed=True)
    result = grouped.agg(count=("fraud", "count"), fraud_count=("fraud", "sum"))
    result["fraud_rate"] = (result["fraud_count"] / result["count"]).round(4)
    return result.reset_index()


def feature_distribution_comparison(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Compare feature means and stds between fraud and legitimate transactions.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.

    Returns
    -------
    pd.DataFrame
        Side-by-side comparison of feature distributions.
    """
    numeric = X.select_dtypes(include=[np.number])
    legit = numeric[y == 0]
    fraud = numeric[y == 1]

    records = []
    for col in numeric.columns:
        records.append({
            "feature": col,
            "legit_mean": round(legit[col].mean(), 4),
            "legit_std": round(legit[col].std(), 4),
            "fraud_mean": round(fraud[col].mean(), 4),
            "fraud_std": round(fraud[col].std(), 4),
            "mean_diff_pct": round(
                abs(fraud[col].mean() - legit[col].mean())
                / max(abs(legit[col].mean()), 1e-6) * 100, 2
            ),
        })
    return pd.DataFrame(records).sort_values("mean_diff_pct", ascending=False)
