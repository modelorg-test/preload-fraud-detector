# Methodology & Assumptions

## Model Architecture

An **XGBoost** gradient boosting ensemble was selected for its ability to handle high-cardinality categorical features and provide native feature importance rankings compatible with regulatory explainability requirements.

### Hyperparameters

| Parameter | Value | Rationale |
| --- | --- | --- |
| `n_estimators` | 500 | Convergence plateau identified at ~450 |
| `max_depth` | 6 | Regularisation to prevent overfitting on rare SAR labels |
| `learning_rate` | 0.05 | Conservative step size for stability |
| `scale_pos_weight` | 83 | Inverse of SAR prevalence (~1.2%) |
| `subsample` | 0.8 | Row-level bagging for variance reduction |

### Explainability

SHAP TreeExplainer is computed for every scored transaction. The top 3 contributing features are surfaced to the investigation desk alongside the suspicion score to satisfy CBUAE explainability requirements.

## Key Assumptions

1. **Label Quality:** Confirmed SARs are assumed to be true positives. Dismissed investigations are treated as true negatives.
2. **Temporal Stability:** Transaction patterns are assumed stable within a 12-month window.
3. **Channel Independence:** The model treats each channel (SWIFT/SEPA) as independent; cross-channel structuring is handled by a separate graph model.
