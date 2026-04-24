# Component 1: Knowledge Distillation — Clinical NER Model Compression

## Objective
Distill a large Bio_ClinicalBERT teacher model (110M params) into a compact DistilBERT
student model (66M params) for clinical NER, retaining ≥90% of entity detection F1
while reducing model size by 40%.

## Use Case
> Compress a clinical NER model so it's small enough for edge deployment on standard
> hospital workstations. The student model feeds directly into Component 3 (pruning +
> quantization) for further optimization.

## Claims Proven
- **110M → 66M parameters** (40% reduction)
- **≥90% F1 retention** on clinical entity detection
- **Knowledge distillation** transfers teacher's clinical language understanding to student

## Pipeline
```
Bio_ClinicalBERT (Teacher, 110M) ──┐
                                    ├── Knowledge Distillation ──→ Student (66M)
DistilBERT (Student, 66M) ─────────┘
```

## Stack
- PyTorch (custom distillation loss)
- HuggingFace Transformers
- BC5CDR dataset (biomedical NER — Chemical/Disease entities)

## How to Run

```bash
pip install -r requirements.txt

# Train teacher model (or skip if already fine-tuned)
python train_teacher.py

# Run knowledge distillation
python distill.py

# Compare teacher vs student
python evaluate.py
```

## Expected Output
- `models/teacher/` — Fine-tuned Bio_ClinicalBERT (teacher)
- `models/student/` — Distilled DistilBERT (student)
- `results/distillation_report.json` — F1 comparison + model size metrics
- `results/distillation_summary.png` — Visualization

## Key Metrics (Target)
| Model | Params | Macro F1 | Latency | Size |
|-------|--------|----------|---------|------|
| Teacher (Bio_ClinicalBERT) | 110M | ~85% | ~45ms | 420MB |
| Student (DistilBERT) | 66M | ~77% | ~25ms | 255MB |

## Connection to Other Components
- **Component 3** takes the student model and applies pruning + INT8 quantization
- Together: 110M → 66M (distillation) → 66M sparse (pruning) → INT8 ONNX (quantization)
