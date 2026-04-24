# Clinical NLP — ML Engineering Portfolio

End-to-end ML engineering portfolio demonstrating model compression, distributed processing,
agentic workflows, statistical evaluation, and production observability — all built around
a clinical NER (Named Entity Recognition) use case for healthcare.

## System Design

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   DATA LAYER                MODEL LAYER              SERVING     │
│                                                                  │
│   PubMed (7K)               Bio_ClinicalBERT         FastAPI     │
│       │                     (Teacher, 110M)           Server     │
│       ▼                          │                      │        │
│   Spark/EMR ──── Weak ──────────▶│                  Prometheus   │
│   (900K docs)    Labels          │                  OpenTelemetry│
│       │                          ▼                  Grafana      │
│   S3 + TF-IDF           DistilClinicalBERT              │       │
│                          (Student, 65M)             Annotation   │
│   Step Functions              │                     UI           │
│   Terraform                   ▼                         │        │
│                          Prune (40%)                    │        │
│                          INT8 (62MB)                    │        │
│                               │                         │        │
│                               ▼                         │        │
│                          A/B Testing ◄──────────────────┘        │
│                          LangChain Agent                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Components

| # | Component | What It Does | Key Tech | Status |
|---|-----------|-------------|----------|--------|
| 1 | [Distillation](distillation/) | Compress Bio_ClinicalBERT (110M) → DistilClinicalBERT (65M) | PyTorch, HuggingFace | ✅ |
| 2 | [Distributed Processing](distributed-training/) | Scale NLP processing + weak labeling on PubMed | PySpark, AWS EMR, Step Functions, Terraform | ✅ |
| 3 | [Model Optimization](model-optimization/) | Prune (40% sparsity) + INT8 quantize for edge deployment | PyTorch, ONNX Runtime, HuggingFace Optimum | ✅ |
| 4 | [Agentic Workflows](agentic-workflows/) | LangChain agent auto-analyzes benchmark reports | LangChain, OpenRouter (Nemotron) | ✅ |
| 5 | [A/B Testing](ab-testing/) | Statistical comparison of model variants | scipy, Mann-Whitney, Wilcoxon | ✅ |
| 6 | [Observability](observability/) | Prometheus metrics, structured logging, OpenTelemetry tracing | FastAPI, Prometheus, OpenTelemetry, Grafana | ✅ |

## Use Case: Clinical NER for PHI Detection

All components are built around a single healthcare NLP use case:
**detecting clinical entities (drugs, diseases) in medical text** — a proxy for
PHI (Protected Health Information) detection used in HIPAA-compliant de-identification.

```
Input:  "Patient was prescribed metformin 500mg for type 2 diabetes"
Output: [metformin 500mg] → Chemical    [type 2 diabetes] → Disease
```

The pipeline optimizes this model for edge deployment — fast enough for real-time
annotation at point of care, small enough for standard workstations, with PHI
never leaving the local device.

## Real Results

### Component 1: Knowledge Distillation

| Metric | Teacher (Bio_ClinicalBERT) | Student v1 (DistilBERT) | Student v2 (DistilClinicalBERT) |
|--------|---------------------------|------------------------|-------------------------------|
| Parameters | 107.7M | 66.4M | 65.2M |
| Macro F1 | 86.57% | 76.06% | 80.70% |
| Chemical F1 | 91.68% | 73.58% | 85.92% |
| Disease F1 | 77.99% | 66.16% | 70.48% |
| Chemical Recall | 92.04% | 63.77% | 82.69% |
| Disease Recall | 81.54% | 61.40% | 69.16% |
| Latency | 39ms | 21ms | 11ms |
| Size | 411MB | 253MB | 249MB |
| F1 Retention | — | 87.9% | 93.2% |

Key finding: Switching from a generic student (DistilBERT) to a clinically pre-trained student
(DistilClinicalBERT) improved Macro F1 by +4.64% and Chemical Recall by +18.92%.
Domain-specific pre-training matters more than hyperparameter tuning.

### Component 2: Distributed Processing (AWS EMR)

| Metric | Value |
|--------|-------|
| Records processed | 899,999 |
| Throughput | 4,350 docs/sec |
| Runtime | 206.9 seconds |
| Cluster | 4× m5.xlarge (EMR) |
| PubMed abstracts labeled (weak labeling) | 7,064 |
| Entities generated (high-confidence) | 19,506 |
| Chemical entities | 9,109 |
| Disease entities | 10,397 |

### Component 3: Model Optimization (Pruning + INT8)

**On Teacher (Bio_ClinicalBERT, 107.7M params):**

| Variant | Macro F1 | F1 Drop | Size | Latency (Mac) |
|---------|----------|---------|------|---------------|
| Baseline (FP32) | 86.57% | — | 411MB | 30ms |
| Pruned (40%) + Recovery | 86.44% | 0.13% | 411MB | 29ms |
| Quantized (INT8) | 85.80% | 0.77% | 103.5MB | 83ms* |

**On Student v1 (DistilBERT, 66.4M params):**

Student v1 uses a different tokenizer than the teacher (uncased vs clinical vocabulary),
causing a tokenizer-model mismatch when evaluated in the optimization pipeline.
Baseline F1 dropped to 17.54% — not because the model is bad (it scores 76.06% with
its own tokenizer), but because the tokens don't align. This confirms that student v1
is incompatible with the teacher's deployment pipeline. Student v2 (same tokenizer
family as teacher) does not have this issue.

**On Student v2 (DistilClinicalBERT, 65.2M params):**

| Variant | Macro F1 | F1 Drop | Size | Latency (Mac) |
|---------|----------|---------|------|---------------|
| Baseline (FP32) | 82.76% | — | 249MB | 16ms |
| Pruned (40%) + Recovery | 83.71% | -0.95% (improved!) | 249MB | 15ms |
| Quantized (INT8) | 75.69% | 7.07% | 62.6MB | 29ms* |

*INT8 latency is slower on macOS ARM — optimized for x86 AVX-512 servers where 3x speedup is expected.

Key finding: INT8 quantization works well on large models (0.77% drop on teacher) but
degrades smaller models significantly (7% drop on student). Pruning with recovery
fine-tuning actually improved the student's F1 beyond baseline.

### Component 5: A/B Testing

**Teacher vs Student v2:**
- Student retains 89.9% of teacher's per-sample F1
- Latency: Teacher 28ms vs Student 29ms (similar on macOS)
- Recommendation: Deploy student — acceptable F1 retention with 40% fewer parameters

**Student v1 vs Student v2:**
- v2 significantly outperforms v1 across all metrics
- Chemical F1: v1 73.58% → v2 85.92% (+12.34%)
- Statistical tests confirm the improvement is real, not noise

### Component 6: Observability
- 101 requests processed, zero errors
- 97% SLA compliance (98/101 requests under 50ms)
- 131 Chemical + 73 Disease entities detected
- Web UI: annotation tool + teacher vs student comparison view

## Data Flow

```
                    ┌──────────────────────────────────────────┐
                    │           DATA PIPELINE                   │
                    │                                           │
  Raw Clinical  ──▶ │  PySpark (Component 2)                   │
  Text (S3)         │  ├── Clean + normalize                   │
                    │  ├── Tokenize                            │
                    │  ├── TF-IDF + N-gram features            │
                    │  └── Write to S3 (Parquet)               │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │           MODEL PIPELINE                  │
                    │                                           │
  BC5CDR Dataset ──▶│  1. Fine-tune Bio_ClinicalBERT (NER)    │
                    │  2. Distill → DistilBERT (Component 1)   │
                    │  3. Prune 40% + INT8 Quantize (Comp 3)   │
                    │  4. Export ONNX for edge deployment       │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │           EVALUATION                      │
                    │                                           │
                    │  5. A/B Test: Teacher vs Optimized        │
                    │     └── Mann-Whitney + Wilcoxon tests     │
                    │  4. Agent auto-analyzes results           │
                    │     └── Reads reports, flags regressions  │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │           PRODUCTION                      │
                    │                                           │
                    │  6. FastAPI inference server              │
                    │     ├── Prometheus metrics (/metrics)     │
                    │     ├── OpenTelemetry tracing             │
                    │     ├── Structured JSON logging           │
                    │     └── Grafana SLA dashboard             │
                    └──────────────────────────────────────────┘
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| ML/DL | PyTorch, HuggingFace Transformers, ONNX Runtime |
| Data | PySpark, Apache Spark, Parquet |
| Cloud | AWS EMR, S3, Step Functions, Terraform |
| Serving | FastAPI, Prometheus, OpenTelemetry, Grafana |
| Evaluation | scipy (Mann-Whitney, Wilcoxon), pandas, matplotlib |
| Agents | LangChain, OpenRouter (Nvidia Nemotron) |
| Models | Bio_ClinicalBERT, DistilBERT, DistilClinicalBERT |
| Dataset | BC5CDR (BioCreative V Chemical Disease Relation), PubMed abstracts |

## Repository Structure

```
clinical-nlp-portfolio/
├── distillation/                 ← Knowledge distillation (Teacher → Student)
│   ├── train_teacher.py          ← Fine-tune Bio_ClinicalBERT
│   ├── distill.py                ← Distill to student v1 (DistilBERT)
│   ├── distill_v2.py             ← Distill to student v2 (DistilClinicalBERT) — best
│   ├── evaluate.py               ← Compare teacher vs student
│   └── UNDERSTANDING_THE_CODE.md
│
├── distributed-training/         ← Distributed NLP pipeline
│   ├── spark_pipeline.py         ← PySpark pipeline (5 stages)
│   ├── pipeline_local.py         ← Local test with synthetic data
│   ├── deploy_emr.py             ← AWS EMR deployment
│   ├── download_pubmed.py        ← Download PubMed abstracts via API
│   ├── weak_label_pubmed.py      ← Teacher NER on PubMed (weak labeling)
│   ├── terraform/                ← Infrastructure as Code
│   └── UNDERSTANDING_THE_CODE.md
│
├── model-optimization/           ← Pruning + INT8 quantization
│   ├── optimize.py               ← Full optimization pipeline
│   ├── benchmark.py              ← Benchmark comparison
│   ├── kv_cache_quantization/    ← PolarQuant (3-bit KV cache)
│   └── UNDERSTANDING_THE_CODE.md
│
├── agentic-workflows/            ← LangChain evaluation agent
│   ├── agent.py                  ← Agent with tool calling
│   ├── tools.py                  ← 5 analysis tools
│   ├── run_tools.py              ← Direct tool runner (no LLM)
│   └── UNDERSTANDING_THE_CODE.md
│
├── ab-testing/                   ← Statistical A/B testing
│   ├── ab_test.py                ← Full experiment + stats
│   └── UNDERSTANDING_THE_CODE.md
│
├── observability/                ← Production monitoring
│   ├── inference_server.py       ← Instrumented FastAPI server
│   ├── metrics.py                ← Prometheus metric definitions
│   ├── logger.py                 ← Structured JSON logging
│   ├── tracing.py                ← OpenTelemetry setup
│   ├── test_client.py            ← Load test client
│   ├── static/index.html         ← Clinical annotation UI
│   ├── static/compare.html       ← Teacher vs Student comparison UI
│   ├── grafana/dashboard.json    ← Grafana dashboard config
│   └── UNDERSTANDING_THE_CODE.md
│
└── README.md                     ← This file
```

## Quick Start

```bash
# Component 1: Knowledge Distillation
cd 01-distillation && pip install -r requirements.txt
python train_teacher.py    # Fine-tune teacher (~15 min)
python distill.py          # Distill to student (~40 min)
python evaluate.py         # Compare results

# Component 2: Distributed Pipeline
cd ../02-distributed-training && pip install -r requirements.txt
python pipeline_local.py   # Local test with synthetic data

# Component 3: Model Optimization
cd ../03-model-optimization && pip install -r requirements.txt
python optimize.py         # Prune + quantize (~20 min)

# Component 4: Agentic Workflows
cd ../04-agentic-workflows && pip install -r requirements.txt
python run_tools.py        # No LLM needed
# or: export OPENROUTER_API_KEY=key && python agent.py

# Component 5: A/B Testing
cd ../05-ab-testing && pip install -r requirements.txt
python ab_test.py --num-samples 100

# Component 6: Observability
cd ../06-observability && pip install -r requirements.txt
python inference_server.py  # Terminal 1
python test_client.py       # Terminal 2
```

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Domain | Clinical NER (healthcare) | Relevant to BCBS, demonstrates PHI awareness |
| Base model | Bio_ClinicalBERT | Pre-trained on clinical text, understands medical language |
| Dataset | BC5CDR | Public biomedical NER, no PHI, same architecture as PHI detection |
| Compression | Distillation → Pruning → INT8 | Three-stage pipeline, each independently valuable |
| Orchestration | AWS Step Functions | Serverless, native EMR integration |
| IaC | Terraform | Version-controlled infrastructure |
| Evaluation | Statistical tests | Not just averages — p-values and confidence intervals |
| Observability | Prometheus + OTel + JSON logs | Industry standard three pillars |
| Agent LLM | Nvidia Nemotron via OpenRouter | Free, healthcare-relevant, tool calling support |
