# Architecture Overview

| Component | Tech |
| --- | --- |
| Ingestion | `pyspark` / Kafka |
| Inference | `xgboost` (15ms p99) |