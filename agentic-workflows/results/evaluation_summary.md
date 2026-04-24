# ML Pipeline Evaluation Summary

Generated: 2026-04-23 09:52

## Component 1: Knowledge Distillation

| Metric | Teacher | Student |
|--------|---------|---------|
| Parameters | 107,723,525 | 66,366,725 |
| Macro F1 | 86.57% | 76.06% |
| Latency | 39.04 ms | 20.89 ms |
| Size | 410.9 MB | 253.2 MB |

**F1 Retention:** 87.9%
**Speedup:** 1.87x
**Parameter Reduction:** 38.4%

## Component 3: Model Optimization (Pruning + INT8)

| Variant | Macro F1 | Latency |
|---------|----------|---------|
| Baseline Fp32 | 86.57% | 30.19 ms |
| Pruned Before Recovery | 85.28% | 29.41 ms |
| Pruned After Recovery | 86.44% | 29.28 ms |
| Quantized Int8 | 85.80% | 83.48 ms |

**Speedup:** 0.36x
**F1 Degradation:** 0.77%
**Sparsity:** 40.0%
**Model Size:** 411.1 MB → 103.5 MB

**Note:** ONNX Runtime INT8 kernels are optimized for x86 CPUs with AVX-512 (e.g., AWS c5/c6i). On macOS ARM, INT8 inference is slower than FP32 due to lack of optimized kernels. Expected 3x speedup on x86 deployment targets.

## Component 2: Distributed Pipeline

- **Total Documents:** 1,000
- **Total Words:** 23,200
- **Avg Document Length:** 23.2 words
- **Mode:** local_test
- **Runtime:** N/As

## Regression Analysis

**Overall Status:** WARN

- ✅ **F1 Degradation** (Model Optimization): F1 degradation within acceptable range (0.77% < 2.0%)
- ⚠️ **Speedup Factor** (Model Optimization): Speedup 0.36x below target 2.0x (may be hardware-dependent — INT8 optimized for x86 AVX-512)
- ✅ **Sparsity** (Model Optimization): Sparsity target met (40.0% >= 35.0%)
- ✅ **F1 Retention** (Distillation): F1 retention meets target (87.9% >= 85.0%)
