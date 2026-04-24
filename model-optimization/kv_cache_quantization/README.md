# KV Cache Quantization: PolarQuant-Inspired 3-Bit Compression

## Overview
Proof-of-concept implementation of sub-4-bit KV cache quantization using polar decomposition
and Johnson-Lindenstrauss random projections. Inspired by TurboQuant (ICLR 2026).

> **Note:** This is an independent implementation based on the published technique description,
> not a reproduction of the official codebase. The core ideas (polar form + JL projection)
> are faithfully implemented, but implementation details may differ from the paper.

## Core Idea
Standard KV cache stores key/value vectors in FP16 (16 bits per element).
At 32K context on a 70B model, that's ~17GB per request.

**PolarQuant approach:**
1. Decompose each KV vector into polar form: magnitude (radius) + direction (angle)
2. The angle distribution is concentrated (known structure) → compress with few bits
3. Apply JL random rotation to collapse remaining dimensions to sign bits (±1)
4. Store: quantized radius (8-bit) + sign bits (1-bit each) + small angle correction

**Result:** ~3 bits/element average → ~5x memory reduction vs FP16

## Architecture
```
KV Vector (FP16, d dims)
    │
    ├── Magnitude: ‖v‖₂ → 8-bit quantized scalar
    │
    └── Direction: v/‖v‖₂
            │
            ├── JL Projection: R·(v/‖v‖₂) where R is (m×d) random matrix
            │       → m sign bits (1-bit each)
            │
            └── Angle residual: top-k correction terms (2-bit each)
            
Storage: 8 bits (radius) + m sign bits + k×2 bits (correction)
Target: m + k chosen so total ≈ 3 bits/element
```

## How to Run
```bash
pip install -r requirements.txt

# Run quantization benchmark
python polar_quant.py

# Run memory comparison
python memory_benchmark.py
```

## Expected Results
| Method | Bits/Element | Memory (32K ctx, 4096 dim) | Cosine Sim | 
|--------|-------------|---------------------------|------------|
| FP16 (baseline) | 16 | 256 MB/layer | 1.000 |
| INT8 | 8 | 128 MB/layer | 0.998 |
| INT4 | 4 | 64 MB/layer | 0.992 |
| PolarQuant (ours) | ~3 | ~48 MB/layer | 0.985+ |
