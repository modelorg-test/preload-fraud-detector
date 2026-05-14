# Limitations & Validation

## Out-of-Time Validation

Independent validation was performed by the Model Validation Unit using 2024 H2 transaction data (1.26M records).

### Performance Metrics

| Metric | Development | OOT (2024 H2) | Threshold | Status |
| --- | --- | --- | --- | --- |
| AUC-ROC | 0.96 | 0.94 | > 0.90 | Pass |
| Precision @ 5% FPR | 0.74 | 0.71 | > 0.65 | Pass |
| Recall @ 5% FPR | 0.88 | 0.85 | > 0.80 | Pass |
| PSI (score dist.) | — | 0.07 | < 0.15 | Pass |

## Known Limitations

1. **Structuring Detection:** The model operates on individual transactions and cannot detect split-payment structuring across multiple transfers. This is addressed by a separate graph-based model.
2. **New Corridors:** Performance degrades for newly opened correspondent banking corridors with < 6 months of historical data.
3. **Sanctions Overlap:** The model does not incorporate sanctions screening — that is handled by a deterministic rules engine upstream.

{{findings:credit_risk}}
