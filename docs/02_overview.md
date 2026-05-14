# Overview & Strategy

## Purpose

Real-time anomaly detection model for SWIFT and SEPA wire transfers. The model evaluates each outbound transaction against historical behavioural patterns to assign a suspicion score used by the Financial Intelligence Unit (FIU) investigation desk.

## Inputs

- `transaction_amount`: Transfer value in AED (float)
- `merchant_category`: MCC code of the counterparty (categorical)
- `user_velocity`: Number of transactions in the trailing 24h window (int)
- `country_risk_score`: FATF risk rating of the destination jurisdiction (float)
- `time_since_last_txn`: Seconds since the customer's previous transaction (float)

## Outputs

- `suspicion_score`: Probability of suspicious activity [0, 1]
- `alert_triggered`: Boolean flag if score exceeds the investigation threshold
- `top_3_features`: SHAP-derived feature attributions for explainability

## Performance Targets

| Metric | Target | Current |
| --- | --- | --- |
| Precision @ 5% FPR | > 0.70 | 0.74 |
| Recall @ 5% FPR | > 0.85 | 0.88 |
| p99 Inference Latency | < 15ms | 12ms |
