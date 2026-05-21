"""Fraud detection model definition using sklearn GradientBoosting.

Builds a pipeline with velocity feature extraction, risk score
binning, and a GradientBoostingClassifier tuned for high-recall
fraud detection at low false-positive rates.
"""

from __future__ import annotations

import logging

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.engineering import (
    RiskScoreBinner,
    VelocityFeatureExtractor,
)

logger = logging.getLogger(__name__)


NUMERIC_COLS = [
    "amount", "amount_log", "distance_from_home",
    "time_since_last_txn", "txn_count_1h", "txn_count_24h",
    "avg_amount_7d", "std_amount_7d", "amount_to_avg_ratio",
]

CATEGORICAL_COLS = [
    "hour_of_day", "day_of_week", "is_weekend", "card_present",
]

RISK_SCORE_COLS = [
    "merchant_risk_score", "country_risk_score",
]


def build_fraud_pipeline(
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    random_state: int = 42,
) -> Pipeline:
    """Build the fraud detection pipeline.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages.
    max_depth : int
        Maximum depth per tree.
    learning_rate : float
        Shrinkage rate.
    subsample : float
        Fraction of samples per tree.
    random_state : int
        Random seed.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Ready-to-fit pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLS),
            ("risk_bins", RiskScoreBinner(score_cols=RISK_SCORE_COLS), RISK_SCORE_COLS),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("velocity", VelocityFeatureExtractor()),
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    min_samples_leaf=50,
                    max_features="sqrt",
                    random_state=random_state,
                ),
            ),
        ]
    )

    logger.info(
        "Built fraud pipeline: n_estimators=%d, max_depth=%d, lr=%.3f",
        n_estimators, max_depth, learning_rate,
    )
    return pipeline
