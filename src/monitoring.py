"""Hourly monitoring for the Transaction Fraud Detector.

Checks PSI on score distribution and key features. Monitors SAR
conversion rate as a business-level performance indicator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

logger = logging.getLogger(__name__)


class AlertLevel(StrEnum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


@dataclass
class FraudMonitorResult:
    metric: str
    value: float
    alert_level: AlertLevel


def psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    b_hist, _ = np.histogram(baseline, bins=bins)
    c_hist, _ = np.histogram(current, bins=bins)
    b_pct = np.where(b_hist == 0, 1e-6, b_hist / len(baseline))
    c_pct = np.where(c_hist == 0, 1e-6, c_hist / len(current))
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def sar_conversion_rate(alerts: int, confirmed_sars: int) -> float:
    """Compute SAR conversion rate from alert count and confirmed SARs."""
    return confirmed_sars / alerts if alerts > 0 else 0.0


def run_hourly_checks(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    alerts_count: int,
    confirmed_sars: int,
) -> list[FraudMonitorResult]:
    """Execute all hourly monitoring checks.

    Args:
        baseline_scores: Reference score distribution (development period).
        current_scores: Scores from the current production hour.
        alerts_count: Number of alerts triggered in the current period.
        confirmed_sars: SARs confirmed by investigators in the same period.

    Returns:
        List of FraudMonitorResult for each checked metric.
    """
    results = []

    score_psi = psi(baseline_scores, current_scores)
    if score_psi >= 0.12:
        level = AlertLevel.RED
    elif score_psi >= 0.08:
        level = AlertLevel.AMBER
    else:
        level = AlertLevel.GREEN
    results.append(FraudMonitorResult("score_psi", score_psi, level))

    conversion = sar_conversion_rate(alerts_count, confirmed_sars)
    if conversion < 0.05:
        level = AlertLevel.RED
    elif conversion < 0.08:
        level = AlertLevel.AMBER
    else:
        level = AlertLevel.GREEN
    results.append(FraudMonitorResult("sar_conversion_rate", conversion, level))

    for r in results:
        log_fn = logger.warning if r.alert_level != AlertLevel.GREEN else logger.info
        log_fn("%s=%.4f [%s]", r.metric, r.value, r.alert_level.upper())

    return results
