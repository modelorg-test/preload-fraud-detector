"""End-to-end training pipeline for the fraud detection model.

Generates a synthetic imbalanced transaction dataset using
sklearn's make_classification, applies SMOTE resampling,
trains the GradientBoosting pipeline, and evaluates with
fraud-specific metrics (precision@FPR, alert volume).

Usage::

    python -m src.pipelines.train
    python -m src.pipelines.train --samples 100000 --fraud-rate 0.012
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.features.engineering import TRANSACTION_FEATURES
from src.models.detector import build_fraud_pipeline
from src.utils.metrics import alert_volume_at_threshold, precision_at_fpr

logger = logging.getLogger(__name__)


def generate_fraud_dataset(
    n_samples: int = 50000,
    fraud_rate: float = 0.012,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic imbalanced transaction dataset.

    Parameters
    ----------
    n_samples : int
        Total number of transactions.
    fraud_rate : float
        Fraction of fraudulent transactions (~1.2% default).
    random_state : int
        Random seed.

    Returns
    -------
    tuple of (X, y)
        X: DataFrame with transaction features.
        y: Series of binary fraud indicators.
    """
    n_features = len(TRANSACTION_FEATURES)

    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 3,
        n_redundant=2,
        n_classes=2,
        weights=[1 - fraud_rate, fraud_rate],
        flip_y=0.005,
        class_sep=1.5,
        random_state=random_state,
    )

    X = pd.DataFrame(X_raw, columns=TRANSACTION_FEATURES)

    # Scale to realistic transaction ranges
    scalers = {
        "amount": (5.0, 5000.0),
        "amount_log": (1.6, 8.5),
        "hour_of_day": (0, 23),
        "day_of_week": (0, 6),
        "is_weekend": (0, 1),
        "merchant_risk_score": (0.0, 1.0),
        "country_risk_score": (0.0, 1.0),
        "card_present": (0, 1),
        "distance_from_home": (0.0, 500.0),
        "time_since_last_txn": (0.0, 720.0),
        "txn_count_1h": (0, 15),
        "txn_count_24h": (0, 50),
        "avg_amount_7d": (10.0, 2000.0),
        "std_amount_7d": (5.0, 800.0),
        "amount_to_avg_ratio": (0.1, 10.0),
    }

    for col, (lo, hi) in scalers.items():
        if col in X.columns:
            col_min, col_max = X[col].min(), X[col].max()
            X[col] = lo + (X[col] - col_min) / (col_max - col_min + 1e-8) * (hi - lo)

    # Ensure integer features
    for int_col in ["hour_of_day", "day_of_week", "is_weekend", "card_present",
                     "txn_count_1h", "txn_count_24h"]:
        if int_col in X.columns:
            X[int_col] = X[int_col].round().clip(lower=0).astype(int)

    y = pd.Series(y, name="is_fraud")

    logger.info(
        "Generated fraud dataset: %d txns, fraud_rate=%.3f%%",
        n_samples, y.mean() * 100,
    )
    return X, y


def train(
    n_samples: int = 50000,
    fraud_rate: float = 0.012,
    test_size: float = 0.20,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Run the full fraud model training workflow.

    Parameters
    ----------
    n_samples : int
        Number of synthetic transactions.
    fraud_rate : float
        Target fraud rate.
    test_size : float
        Holdout fraction for evaluation.
    cv_folds : int
        Cross-validation folds.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Training results.
    """
    X, y = generate_fraud_dataset(
        n_samples=n_samples,
        fraud_rate=fraud_rate,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    logger.info(
        "Split: train=%d (fraud=%.3f%%), test=%d (fraud=%.3f%%)",
        len(X_train), y_train.mean() * 100,
        len(X_test), y_test.mean() * 100,
    )

    pipeline = build_fraud_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    # CV predictions
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    y_prob_cv = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba",
    )[:, 1]

    # Test predictions
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    cv_metrics = {
        "auc_roc": roc_auc_score(y_train, y_prob_cv),
        "avg_precision": average_precision_score(y_train, y_prob_cv),
        "precision_at_1pct_fpr": precision_at_fpr(y_train.values, y_prob_cv, target_fpr=0.01),
    }

    test_metrics = {
        "auc_roc": roc_auc_score(y_test, y_prob_test),
        "avg_precision": average_precision_score(y_test, y_prob_test),
        "precision_at_1pct_fpr": precision_at_fpr(y_test.values, y_prob_test, target_fpr=0.01),
        "alert_volume_at_50pct_recall": alert_volume_at_threshold(
            y_test.values, y_prob_test, recall_target=0.50,
        ),
    }

    # Feature importance
    clf = pipeline.named_steps["classifier"]
    importance = dict(
        zip(
            TRANSACTION_FEATURES,
            clf.feature_importances_[:len(TRANSACTION_FEATURES)],
        )
    )

    logger.info("CV metrics: %s", {k: round(v, 4) for k, v in cv_metrics.items()})
    logger.info("Test metrics: %s", {k: round(v, 4) for k, v in test_metrics.items()})

    return {
        "pipeline": pipeline,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "feature_importance": importance,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--fraud-rate", type=float, default=0.012)
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    results = train(
        n_samples=args.samples,
        fraud_rate=args.fraud_rate,
        test_size=args.test_size,
    )

    print("\n" + "=" * 60)
    print("FRAUD MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nCV AUC-ROC:          {results['cv_metrics']['auc_roc']:.4f}")
    print(f"CV Avg Precision:    {results['cv_metrics']['avg_precision']:.4f}")
    print(f"Test AUC-ROC:        {results['test_metrics']['auc_roc']:.4f}")
    print(f"Test Precision@1%FPR:{results['test_metrics']['precision_at_1pct_fpr']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
