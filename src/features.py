"""Feature engineering for the Transaction Fraud Detector.

Constructs velocity metrics, country risk scores, and temporal
features from raw transaction events for real-time scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FATF grey-list jurisdictions (country risk score 0.8+)
FATF_GREY_LIST: frozenset[str] = frozenset({
    "BF", "CM", "CD", "HT", "IR", "KP", "ML", "MM", "MZ", "NG",
    "PK", "PH", "SA", "SN", "SS", "SY", "TZ", "VN", "YE",
})

COUNTRY_RISK_DEFAULT = 0.3


def country_risk_score(iso2: str) -> float:
    """Map a 2-letter ISO country code to a FATF risk score.

    Args:
        iso2: 2-letter ISO 3166-1 alpha-2 country code.

    Returns:
        Risk score in [0, 1]. Higher = more risk.
    """
    return 0.9 if iso2.upper() in FATF_GREY_LIST else COUNTRY_RISK_DEFAULT


def build_feature_vector(
    txn: pd.Series,
    velocity_window: pd.DataFrame,
) -> dict[str, float]:
    """Build the feature vector for a single transaction.

    Args:
        txn: Single transaction row with at minimum:
             amount_aed, mcc, counterparty_country, timestamp_utc.
        velocity_window: Trailing 24h transactions for the same account.

    Returns:
        Dict of named features ready for XGBoost inference.
    """
    now = pd.Timestamp(txn["timestamp_utc"], tz="UTC")
    last_txn_ts = velocity_window["timestamp_utc"].max() if not velocity_window.empty else None

    time_since_last = (
        (now - last_txn_ts).total_seconds()
        if last_txn_ts is not None
        else 86400.0
    )

    return {
        "transaction_amount": float(txn["amount_aed"]),
        "merchant_category": int(txn.get("mcc", 0)),
        "user_velocity": len(velocity_window),
        "country_risk_score": country_risk_score(txn.get("counterparty_country", "AE")),
        "time_since_last_txn": float(time_since_last),
        "hour_of_day": now.hour,
        "is_weekend": int(now.weekday() >= 5),
    }
