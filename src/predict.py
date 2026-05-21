"""Real-time inference for the Transaction Fraud Detector.

Designed for sub-15ms end-to-end latency. Loads the ONNX model once at
startup and evaluates each payment synchronously inside the scoring API.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import shap

from features import FEATURE_COLS, build_feature_vector

logger = logging.getLogger(__name__)

ALERT_THRESHOLD = 0.65
LATENCY_BUDGET_MS = 15.0


class FraudPredictor:
    """ONNX-backed real-time fraud scoring engine."""

    def __init__(self, model_path: str, xgb_model=None) -> None:
        self._session = ort.InferenceSession(model_path)
        self._input_name = self._session.get_inputs()[0].name
        # SHAP explainer requires the original XGBoost model for TreeExplainer
        self._explainer = shap.TreeExplainer(xgb_model) if xgb_model else None
        logger.info("Loaded ONNX fraud model from %s", model_path)

    def score(
        self,
        txn: pd.Series,
        velocity_window: pd.DataFrame,
    ) -> dict:
        """Score a single transaction against the fraud model.

        Args:
            txn: Raw transaction series.
            velocity_window: Trailing 24h transactions for the account.

        Returns:
            Dict with suspicion_score, alert_triggered, and top_3_features.
        """
        t0 = time.perf_counter()

        features = build_feature_vector(txn, velocity_window)
        X = np.array([[features[c] for c in FEATURE_COLS]], dtype=np.float32)
        (probs,) = self._session.run(None, {self._input_name: X})
        score = float(probs[0, 1])

        top_features: list[str] = []
        if self._explainer is not None:
            shap_vals = self._explainer.shap_values(X)[0]
            top_indices = np.argsort(np.abs(shap_vals))[-3:][::-1]
            top_features = [FEATURE_COLS[i] for i in top_indices]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > LATENCY_BUDGET_MS:
            logger.warning("Scoring latency %.1fms exceeded budget of %.1fms", elapsed_ms, LATENCY_BUDGET_MS)

        return {
            "suspicion_score": score,
            "alert_triggered": score >= ALERT_THRESHOLD,
            "top_3_features": top_features,
            "latency_ms": round(elapsed_ms, 2),
        }
