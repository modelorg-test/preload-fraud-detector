"""Velocity and aggregation feature engineering for fraud detection.

Computes transaction-level features that capture spending patterns,
velocity anomalies, and merchant-category risk indicators. All
transformers follow the sklearn fit/transform API.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


# Synthetic feature name mapping for make_classification output
TRANSACTION_FEATURES = [
    "amount",
    "amount_log",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "merchant_risk_score",
    "country_risk_score",
    "card_present",
    "distance_from_home",
    "time_since_last_txn",
    "txn_count_1h",
    "txn_count_24h",
    "avg_amount_7d",
    "std_amount_7d",
    "amount_to_avg_ratio",
]


class VelocityFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract velocity-based features from raw transaction data.

    Computes rolling aggregations and ratio features that capture
    anomalous spending patterns relative to the cardholder's baseline.

    Parameters
    ----------
    amount_col : str
        Column name for the transaction amount.
    avg_col : str
        Column name for the 7-day average amount.
    std_col : str
        Column name for the 7-day standard deviation.
    """

    def __init__(
        self,
        amount_col: str = "amount",
        avg_col: str = "avg_amount_7d",
        std_col: str = "std_amount_7d",
    ) -> None:
        self.amount_col = amount_col
        self.avg_col = avg_col
        self.std_col = std_col

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> VelocityFeatureExtractor:
        """Fit is a no-op — velocity features are stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute derived velocity features.

        Parameters
        ----------
        X : pd.DataFrame
            Transaction-level DataFrame.

        Returns
        -------
        pd.DataFrame
            Original columns plus derived velocity features.
        """
        result = X.copy()

        # Amount z-score relative to cardholder baseline
        if self.avg_col in result.columns and self.std_col in result.columns:
            std_safe = result[self.std_col].clip(lower=1e-6)
            result["amount_zscore"] = (
                (result[self.amount_col] - result[self.avg_col]) / std_safe
            )

        # Log-amount if not already present
        if "amount_log" not in result.columns and self.amount_col in result.columns:
            result["amount_log"] = np.log1p(result[self.amount_col].clip(lower=0))

        # Velocity burst indicator: > 3 txns in 1 hour
        if "txn_count_1h" in result.columns:
            result["velocity_burst"] = (result["txn_count_1h"] > 3).astype(int)

        # High-risk hours (00:00–05:00)
        if "hour_of_day" in result.columns:
            result["is_nighttime"] = (result["hour_of_day"] < 5).astype(int)

        return result


class RiskScoreBinner(BaseEstimator, TransformerMixin):
    """Bin continuous risk scores into categorical risk tiers.

    Parameters
    ----------
    score_cols : list[str]
        Column names containing risk scores to bin.
    n_bins : int
        Number of risk tier bins.
    """

    def __init__(
        self,
        score_cols: list[str] | None = None,
        n_bins: int = 5,
    ) -> None:
        self.score_cols = score_cols or ["merchant_risk_score", "country_risk_score"]
        self.n_bins = n_bins

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RiskScoreBinner:
        """Compute bin edges from training data."""
        self.bin_edges_: dict[str, np.ndarray] = {}
        for col in self.score_cols:
            if col in X.columns:
                _, edges = pd.qcut(
                    X[col], q=self.n_bins, retbins=True, duplicates="drop"
                )
                self.bin_edges_[col] = edges
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply binning to risk score columns."""
        result = X.copy()
        for col, edges in self.bin_edges_.items():
            if col in result.columns:
                result[f"{col}_tier"] = pd.cut(
                    result[col],
                    bins=edges,
                    labels=False,
                    include_lowest=True,
                ).fillna(0).astype(int)
        return result
