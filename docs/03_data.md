# Data Set Description

## Training Dataset

The model was trained on **8.4 million** wire transfer records spanning January 2021 to December 2024, sourced from the core payment processing system.

### Composition by Channel

| Channel | Records | SAR Rate | Notes |
| --- | --- | --- | --- |
| SWIFT MT103 | 3,200,000 | 0.12% | Cross-border individual transfers |
| SWIFT MT202 | 1,800,000 | 0.08% | Bank-to-bank cover payments |
| SEPA Credit | 2,400,000 | 0.15% | Intra-EEA transfers |
| Internal Transfers | 1,000,000 | 0.03% | Domestic same-bank movements |

### Exclusions

- Transactions below AED 1,000 (below reporting threshold)
- Batch salary payments (structurally distinct from individual transfers)
- Sanctions-screened blocked transactions (handled by a separate rules engine)

### Label Source

Ground truth labels are derived from confirmed SAR filings submitted to the UAE FIU, with a 6-month lookback window applied to account for investigation lag.

### Data Split

| Split | Records | Period |
| --- | --- | --- |
| Training | 5,880,000 (70%) | 2021–2023 |
| Validation | 1,260,000 (15%) | 2024 H1 |
| Out-of-Time Test | 1,260,000 (15%) | 2024 H2 |
