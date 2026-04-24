# Student v1 vs v2: Side-by-Side Comparison

## Distillation Config

| Parameter | v1 | v2 |
|-----------|----|----|
| Epochs | 5 | 10 |
| Temperature | 4.0 | 4.0 |
| Alpha (soft weight) | 0.5 | 0.5 |
| Learning Rate | 5e-5 | 5e-05 |
| LR Warmup | None | 10% |

## Model Metrics

| Metric | Teacher | Student v1 | Student v2 | v1→v2 Change |
|--------|---------|-----------|-----------|-------------|
| Macro F1 | 86.57% | 76.06% | 80.70% | +4.64% |
| Weighted F1 | 95.40% | 91.56% | 93.09% | +1.53% |
| Chemical F1 | 91.68% | 73.58% | 85.92% | +12.34% |
| Disease F1 | 77.99% | 66.16% | 70.48% | +4.32% |
| Chemical Precision | 91.32% | 86.95% | 89.40% | +2.45% |
| Chemical Recall | 92.04% | 63.77% | 82.69% | +18.92% |
| Disease Precision | 74.73% | 71.73% | 71.86% | +0.13% |
| Disease Recall | 81.54% | 61.40% | 69.16% | +7.76% |
| Parameters | 107,723,525 | 65,196,293 | 65,196,293 | — |
| Model Size | 410.9 MB | 248.7 MB | 248.7 MB | — |
| Latency (mean) | 20.30 | 10.69 | 10.77 | +0.08 |
| Latency (P95) | 21.65 | 11.28 | 10.94 | -0.34 |

## Summary

- **Macro F1 improvement:** +4.64%
- **Chemical F1 improvement:** +12.34%
- **Disease F1 improvement:** +4.32%
- **F1 retention (v1):** 87.9% of teacher
- **F1 retention (v2):** 93.2% of teacher