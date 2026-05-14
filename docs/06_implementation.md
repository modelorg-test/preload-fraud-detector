# Implementation Process

## Real-Time Scoring

The fraud detection model is deployed as a sidecar service within the payment processing pipeline. Every outbound wire transfer is scored before settlement authorisation.

### Latency Budget

| Stage | Target | Measured |
| --- | --- | --- |
| Feature retrieval (Redis) | < 3ms | 2.1ms |
| SHAP computation | < 5ms | 4.3ms |
| Model inference (XGBoost) | < 5ms | 3.8ms |
| Total end-to-end | < 15ms | 12ms |

{{stage:credit_risk}}

{{experiment:credit_risk}}
