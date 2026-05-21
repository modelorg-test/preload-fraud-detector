# Fraud Detection Model

Real-time transaction fraud scoring using Gradient Boosting with
SMOTE-based resampling for class imbalance.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.pipelines.train
```

## Directory Structure

```
src/
├── models/          # GradientBoosting pipeline definition
├── pipelines/       # Training and evaluation entrypoints
├── eda/             # Transaction pattern analysis
├── features/        # Velocity and aggregation features
├── utils/           # Alert metrics, SHAP explanations
└── notebooks/       # Exploration notebooks
```
