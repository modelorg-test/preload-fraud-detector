# Ongoing Monitoring

## Monitoring Framework

The model is monitored **hourly** using PSI on the score distribution and key input features. Automated alerts are routed to the Fraud Analytics on-call team.

### Alert Thresholds

| Metric | Green | Amber | Red |
| --- | --- | --- | --- |
| PSI (Score Distribution) | < 0.08 | 0.08–0.12 | > 0.12 |
| PSI (Feature Drift) | < 0.10 | 0.10–0.20 | > 0.20 |
| Alert Volume Shift | < 15% | 15–30% | > 30% |
| SAR Conversion Rate | > 8% | 5–8% | < 5% |

### Escalation Protocol

- **Green:** No action. Hourly metrics logged to dashboard.
- **Amber:** On-call analyst investigates within 2 hours. Root cause documented.
- **Red:** Immediate escalation to MLRO. Model may be suspended; fallback to rules-only scoring.

{{baseline:credit_risk}}

{{monitor:credit_risk}}
