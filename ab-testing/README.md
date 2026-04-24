# Component 5: A/B Testing Framework — Model Evaluation

## Objective
Build an A/B testing harness that splits inference traffic between the teacher (Model A)
and optimized model (Model B), logs predictions + latency per variant, runs statistical
significance tests, and outputs a winner with confidence intervals.

## Why A/B Testing?
Benchmarks on a fixed test set don't tell the full story. A/B testing answers:
"Does the optimized model actually perform as well on varied, real-world-like data?"

It catches issues benchmarks miss:
- Performance degradation on specific text patterns
- Latency spikes on long documents
- Entity types where the optimized model is weaker

## How It Works
```
Clinical Text → Router (50/50 split) → Model A (Teacher) + Model B (Optimized)
                                              ↓                    ↓
                                        Predictions + Latency logged per variant
                                              ↓                    ↓
                                        Statistical Tests (Mann-Whitney, McNemar's)
                                              ↓
                                        Winner + Confidence Interval
```

## Stack
- Python (scipy for statistical tests)
- pandas (experiment logging and analysis)
- matplotlib (visualization)
- MLflow (optional — experiment tracking)

## How to Run
```bash
pip install -r requirements.txt
python ab_test.py                    # Run full A/B experiment
python ab_test.py --num-samples 500  # Custom sample size
```

## Output
- `results/ab_test_report.json` — Statistical test results
- `results/ab_test_summary.png` — Visualization (latency + accuracy comparison)
- `results/experiment_log.csv` — Raw per-sample predictions and latencies
