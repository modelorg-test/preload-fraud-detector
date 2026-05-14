# Expert Judgement & Impact

## Investigation Threshold Calibration

The alert threshold (suspicion_score > 0.65) was calibrated jointly by the Fraud Analytics team and the FIU investigation desk to balance detection rate against investigator capacity.

### Threshold Analysis

| Threshold | Precision | Recall | Daily Alerts | Investigator Capacity |
| --- | --- | --- | --- | --- |
| 0.50 | 0.58 | 0.94 | 320 | Exceeds capacity |
| 0.60 | 0.68 | 0.90 | 185 | Manageable |
| **0.65** | **0.74** | **0.88** | **140** | **Selected** |
| 0.70 | 0.79 | 0.82 | 95 | Under-utilised |
| 0.80 | 0.88 | 0.71 | 48 | Unacceptable recall |

## Override Policy

Manual threshold overrides for specific jurisdictions or MCC categories require documented approval from the MLRO (Money Laundering Reporting Officer).
