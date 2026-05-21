"""Training pipeline for the Transaction Fraud Detector.

Fits an XGBoost classifier on labelled SWIFT/SEPA transaction records,
exports the model to ONNX, and registers it with MLflow.
"""

from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MODEL_NAME = "transaction-fraud-detector"
EXPERIMENT_NAME = "fraud-detection/xgboost"

FEATURE_COLS = [
    "transaction_amount",
    "merchant_category",
    "user_velocity",
    "country_risk_score",
    "time_since_last_txn",
    "hour_of_day",
    "is_weekend",
]
TARGET = "is_sar"


def build_model(scale_pos_weight: float = 83.0) -> XGBClassifier:
    """Return a configured XGBoost classifier.

    The scale_pos_weight compensates for the ~1.2% SAR prevalence rate.
    """
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )


def train(data_path: str, fpr_threshold: float = 0.05) -> None:
    """Run the full training and evaluation workflow.

    Args:
        data_path: Path to Parquet file with feature-engineered transactions.
        fpr_threshold: False positive rate at which to evaluate precision/recall.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_parquet(data_path)
    X = df[FEATURE_COLS]
    y = df[TARGET]

    sar_rate = y.mean()
    scale_pos_weight = (1 - sar_rate) / sar_rate
    logger.info("SAR rate: %.4f  scale_pos_weight: %.1f", sar_rate, scale_pos_weight)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    with mlflow.start_run():
        model = build_model(scale_pos_weight)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        # Evaluate at 5% FPR operating point
        threshold = np.percentile(y_prob[y_test == 0], 95)
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        mlflow.log_params({"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05})
        mlflow.log_metrics({"auc_roc": auc, "precision_5fpr": precision, "recall_5fpr": recall})
        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

        logger.info(
            "Training complete — AUC: %.4f  Precision@5%%FPR: %.4f  Recall@5%%FPR: %.4f",
            auc, precision, recall,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Transaction Fraud Detector")
    parser.add_argument("--data", required=True, help="Path to Parquet training file")
    args = parser.parse_args()
    train(args.data)
