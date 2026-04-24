import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

TEACHER_MODEL_PATH = Path("models/teacher")
STUDENT_BASE = "nlpie/distil-clinicalbert"
STUDENT_V1_PATH = Path("models/student")
STUDENT_V2_PATH = Path("models/student_v2")
RESULTS_DIR = Path("results")

BATCH_SIZE = 16
MAX_SEQ_LENGTH = 128
SEED = 42

NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
TEMPERATURE = 4.0
ALPHA = 0.5
WARMUP_RATIO = 0.1
BATCH_SIZE = 16
PATIENCE = 3

LABEL_LIST = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def load_data(tokenizer):
    dataset = load_dataset("tner/bc5cdr")

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            is_split_into_words=True,
        )
        all_labels = []
        for i, labels in enumerate(examples["tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[word_idx])
                else:
                    tag = labels[word_idx]
                    if tag == 1:
                        label_ids.append(2)
                    elif tag == 3:
                        label_ids.append(4)
                    else:
                        label_ids.append(tag)
                previous_word_idx = word_idx
            all_labels.append(label_ids)
        tokenized["labels"] = all_labels
        return tokenized

    tokenized = dataset.map(tokenize_and_align, batched=True,
                            remove_columns=dataset["train"].column_names)
    tokenized.set_format("torch")
    return tokenized


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    mask = (labels != -100).unsqueeze(-1).expand_as(student_logits)

    soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    soft_loss = F.kl_div(
        soft_student.masked_select(mask).view(-1, student_logits.size(-1)),
        soft_teacher.masked_select(mask).view(-1, teacher_logits.size(-1)),
        log_target=True,
        reduction="batchmean",
    ) * (temperature ** 2)

    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    return alpha * soft_loss + (1 - alpha) * hard_loss


def evaluate_ner(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            for pred_seq, label_seq in zip(preds, labels.numpy()):
                for p, l in zip(pred_seq, label_seq):
                    if l != -100:
                        all_preds.append(p)
                        all_labels.append(l)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(len(LABEL_LIST))),
        target_names=LABEL_LIST,
        output_dict=True,
        zero_division=0,
    )

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "chemical_f1": round(report.get("B-Chemical", {}).get("f1-score", 0), 4),
        "disease_f1": round(report.get("B-Disease", {}).get("f1-score", 0), 4),
        "chemical_precision": round(report.get("B-Chemical", {}).get("precision", 0), 4),
        "chemical_recall": round(report.get("B-Chemical", {}).get("recall", 0), 4),
        "disease_precision": round(report.get("B-Disease", {}).get("precision", 0), 4),
        "disease_recall": round(report.get("B-Disease", {}).get("recall", 0), 4),
    }


def benchmark_latency(model, tokenizer, device, num_runs=50):
    model.eval()
    model.to(device)
    text = "Patient was prescribed metformin 500mg for type 2 diabetes mellitus."
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        for _ in range(5):
            model(**inputs)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(**inputs)
            latencies.append((time.perf_counter() - start) * 1000)

    return {
        "latency_mean_ms": round(np.mean(latencies), 2),
        "latency_p50_ms": round(np.percentile(latencies, 50), 2),
        "latency_p95_ms": round(np.percentile(latencies, 95), 2),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STUDENT_V2_PATH.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Distillation v2: epochs={NUM_EPOCHS}, T={TEMPERATURE}, "
          f"alpha={ALPHA}, lr={LEARNING_RATE}")
    print(f"Device: {device}")

    print("\n[1/6] Loading teacher...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH)
    teacher = AutoModelForTokenClassification.from_pretrained(TEACHER_MODEL_PATH)
    teacher.to(device)
    teacher.eval()
    teacher_params = sum(p.numel() for p in teacher.parameters())

    print("[2/6] Initializing student v2...")
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE)
    student = AutoModelForTokenClassification.from_pretrained(
        STUDENT_BASE, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID,
    )
    student.to(device)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"  Teacher: {teacher_params:,} params")
    print(f"  Student: {student_params:,} params")

    student.gradient_checkpointing_enable()

    print("[3/6] Loading data...")
    teacher_data = load_data(teacher_tokenizer)
    student_data = load_data(student_tokenizer)

    train_teacher_loader = DataLoader(teacher_data["train"], batch_size=BATCH_SIZE, shuffle=False)
    train_student_loader = DataLoader(student_data["train"], batch_size=BATCH_SIZE, shuffle=False)
    val_student_loader = DataLoader(student_data["validation"], batch_size=BATCH_SIZE)
    test_student_loader = DataLoader(student_data["test"], batch_size=BATCH_SIZE)

    print("[4/6] Starting distillation v2...")
    total_steps = NUM_EPOCHS * len(train_student_loader)
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # cosine schedule with linear warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1 = 0.0
    history = []
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        student.train()
        if hasattr(student, 'gradient_checkpointing_disable'):
            student.gradient_checkpointing_enable()
        
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            zip(train_teacher_loader, train_student_loader),
            total=len(train_student_loader),
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
        )

        for teacher_batch, student_batch in pbar:
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=teacher_batch["input_ids"].to(device),
                    attention_mask=teacher_batch["attention_mask"].to(device),
                )

            student_outputs = student(
                input_ids=student_batch["input_ids"].to(device),
                attention_mask=student_batch["attention_mask"].to(device),
            )
            labels = student_batch["labels"].to(device)

            min_seq = min(teacher_outputs.logits.size(1), student_outputs.logits.size(1))
            loss = distillation_loss(
                student_outputs.logits[:, :min_seq, :],
                teacher_outputs.logits[:, :min_seq, :],
                labels[:, :min_seq],
                temperature=TEMPERATURE, alpha=ALPHA,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / num_batches
        val_metrics = evaluate_ner(student, val_student_loader, device)

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_f1={val_metrics['macro_f1']*100:.2f}%")

        history.append({
            "epoch": epoch + 1,
            "loss": round(avg_loss, 4),
            "val_macro_f1": val_metrics["macro_f1"],
        })

        if val_metrics["macro_f1"] > best_f1 + 0.001:
            best_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            student.save_pretrained(STUDENT_V2_PATH)
            student_tokenizer.save_pretrained(STUDENT_V2_PATH)
            print(f"  New best! F1: {best_f1*100:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print("\n[5/6] Evaluating all models on test set...")

    student_v2 = AutoModelForTokenClassification.from_pretrained(STUDENT_V2_PATH)
    student_v2.to(device)

    teacher_test_loader = DataLoader(teacher_data["test"], batch_size=BATCH_SIZE)
    teacher_data["test"].set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    teacher_metrics = evaluate_ner(teacher, teacher_test_loader, device)
    teacher_latency = benchmark_latency(teacher, teacher_tokenizer, device)

    v1_metrics = None
    v1_latency = None
    if STUDENT_V1_PATH.exists():
        student_v1 = AutoModelForTokenClassification.from_pretrained(STUDENT_V1_PATH)
        student_v1.to(device)
        v1_tokenizer = AutoTokenizer.from_pretrained(STUDENT_V1_PATH)
        v1_data = load_data(v1_tokenizer)
        v1_test_loader = DataLoader(v1_data["test"], batch_size=BATCH_SIZE)
        v1_metrics = evaluate_ner(student_v1, v1_test_loader, device)
        v1_latency = benchmark_latency(student_v1, v1_tokenizer, device)
        del student_v1

    v2_metrics = evaluate_ner(student_v2, test_student_loader, device)
    v2_latency = benchmark_latency(student_v2, student_tokenizer, device)

    teacher_size = sum(p.nelement() * p.element_size() for p in teacher.parameters()) / (1024*1024)
    v2_size = sum(p.nelement() * p.element_size() for p in student_v2.parameters()) / (1024*1024)

    print("\n[6/6] Generating comparison report...")

    comparison = {
        "title": "Student v1 vs v2 Side-by-Side Comparison",
        "distillation_config": {
            "v1": {"epochs": 5, "temperature": 4.0, "alpha": 0.5, "lr": 5e-5, "warmup": "none"},
            "v2": {"epochs": NUM_EPOCHS, "temperature": TEMPERATURE, "alpha": ALPHA, "lr": LEARNING_RATE, "warmup": f"{WARMUP_RATIO*100:.0f}%"},
        },
        "teacher": {
            "model": "Bio_ClinicalBERT",
            "parameters": teacher_params,
            "model_size_mb": round(teacher_size, 1),
            **teacher_metrics,
            **teacher_latency,
        },
        "student_v1": {
            "model": "DistilBERT (v1)",
            "parameters": student_params,
            "model_size_mb": round(v2_size, 1),
            **(v1_metrics or {}),
            **(v1_latency or {}),
        },
        "student_v2": {
            "model": f"DistilBERT (v2 — {NUM_EPOCHS} epochs, T={TEMPERATURE}, α={ALPHA})",
            "parameters": student_params,
            "model_size_mb": round(v2_size, 1),
            **v2_metrics,
            **v2_latency,
        },
        "improvement": {},
        "training_history_v2": history,
    }

    if v1_metrics:
        comparison["improvement"] = {
            "macro_f1_delta": round((v2_metrics["macro_f1"] - v1_metrics["macro_f1"]) * 100, 2),
            "chemical_f1_delta": round((v2_metrics["chemical_f1"] - v1_metrics["chemical_f1"]) * 100, 2),
            "disease_f1_delta": round((v2_metrics["disease_f1"] - v1_metrics["disease_f1"]) * 100, 2),
            "f1_retention_v1": round(v1_metrics["macro_f1"] / teacher_metrics["macro_f1"] * 100, 1),
            "f1_retention_v2": round(v2_metrics["macro_f1"] / teacher_metrics["macro_f1"] * 100, 1),
        }

    student_name = STUDENT_BASE.replace("/", "_")
    json_file = RESULTS_DIR / f"v1_vs_v2_comparison_{student_name}.json"
    md_file = RESULTS_DIR / f"v1_vs_v2_comparison_{student_name}.md"

    with open(json_file, "w") as f:
        json.dump(comparison, f, indent=2)

    md = []
    md.append(f"# Student v1 vs v2: Side-by-Side Comparison")
    md.append(f"## Student v2 Model: `{STUDENT_BASE}`\n")
    md.append("## Distillation Config\n")
    md.append("| Parameter | v1 | v2 |")
    md.append("|-----------|----|----|")
    md.append(f"| Epochs | 5 | {NUM_EPOCHS} |")
    md.append(f"| Temperature | 4.0 | {TEMPERATURE} |")
    md.append(f"| Alpha (soft weight) | 0.5 | {ALPHA} |")
    md.append(f"| Learning Rate | 5e-5 | {LEARNING_RATE} |")
    md.append(f"| LR Warmup | None | {WARMUP_RATIO*100:.0f}% |")
    md.append("")

    md.append("## Model Metrics\n")
    md.append("| Metric | Teacher | Student v1 | Student v2 | v1->v2 Change |")
    md.append("|--------|---------|-----------|-----------|-------------|")

    def row(name, t_key, fmt=".2f", pct=True):
        t_val = teacher_metrics.get(t_key, teacher_latency.get(t_key, 0))
        v1_val = (v1_metrics or {}).get(t_key, (v1_latency or {}).get(t_key, 0))
        v2_val = v2_metrics.get(t_key, v2_latency.get(t_key, 0))
        if pct:
            delta = (v2_val - v1_val) * 100
            return f"| {name} | {t_val*100:{fmt}}% | {v1_val*100:{fmt}}% | {v2_val*100:{fmt}}% | {'+' if delta >= 0 else ''}{delta:{fmt}}% |"
        else:
            delta = v2_val - v1_val
            return f"| {name} | {t_val:{fmt}} | {v1_val:{fmt}} | {v2_val:{fmt}} | {'+' if delta >= 0 else ''}{delta:{fmt}} |"

    md.append(row("Macro F1", "macro_f1"))
    md.append(row("Weighted F1", "weighted_f1"))
    md.append(row("Chemical F1", "chemical_f1"))
    md.append(row("Disease F1", "disease_f1"))
    md.append(row("Chemical Precision", "chemical_precision"))
    md.append(row("Chemical Recall", "chemical_recall"))
    md.append(row("Disease Precision", "disease_precision"))
    md.append(row("Disease Recall", "disease_recall"))
    md.append(f"| Parameters | {teacher_params:,} | {student_params:,} | {student_params:,} | — |")
    md.append(f"| Model Size | {teacher_size:.1f} MB | {v2_size:.1f} MB | {v2_size:.1f} MB | — |")
    md.append(row("Latency (mean)", "latency_mean_ms", pct=False))
    md.append(row("Latency (P95)", "latency_p95_ms", pct=False))
    md.append("")

    if v1_metrics:
        imp = comparison["improvement"]
        md.append("## Summary\n")
        md.append(f"- **Macro F1 improvement:** {'+' if imp['macro_f1_delta'] >= 0 else ''}{imp['macro_f1_delta']:.2f}%")
        md.append(f"- **Chemical F1 improvement:** {'+' if imp['chemical_f1_delta'] >= 0 else ''}{imp['chemical_f1_delta']:.2f}%")
        md.append(f"- **Disease F1 improvement:** {'+' if imp['disease_f1_delta'] >= 0 else ''}{imp['disease_f1_delta']:.2f}%")
        md.append(f"- **F1 retention (v1):** {imp['f1_retention_v1']:.1f}% of teacher")
        md.append(f"- **F1 retention (v2):** {imp['f1_retention_v2']:.1f}% of teacher")

    md_text = "\n".join(md)
    with open(md_file, "w") as f:
        f.write(md_text)

    print("\n" + md_text)
    print(f"\n  JSON: {json_file}")
    print(f"  Markdown: {md_file}")


if __name__ == "__main__":
    main()
