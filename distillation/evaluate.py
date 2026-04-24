import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report
import warnings

warnings.filterwarnings("ignore")

TEACHER_DIR = Path("models/teacher")
STUDENT_DIR = Path("models/student")
RESULTS_DIR = Path("results")
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
NUM_WARMUP = 5
NUM_LATENCY_RUNS = 50

LABEL_LIST = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]


def load_test_data(tokenizer):
    dataset = load_dataset("tner/bc5cdr", split="test")

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

    dataset = dataset.map(tokenize_and_align, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def benchmark_model(model, dataset, device="cpu"):
    """F1 + latency profiling."""
    model.eval()
    model.to(device)

    sample = dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            model(input_ids=input_ids, attention_mask=attention_mask)

    latencies = []
    with torch.no_grad():
        for _ in range(NUM_LATENCY_RUNS):
            start = time.perf_counter()
            model(input_ids=input_ids, attention_mask=attention_mask)
            latencies.append((time.perf_counter() - start) * 1000)

    all_preds, all_labels = [], []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            for pred_seq, label_seq in zip(preds, labels.numpy()):
                for p, l in zip(pred_seq, label_seq):
                    if l != -100:
                        all_preds.append(p)
                        all_labels.append(l)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(len(LABEL_LIST))),
        target_names=LABEL_LIST,
        output_dict=True,
        zero_division=0,
    )

    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)

    return {
        "parameters": num_params,
        "model_size_mb": round(model_size_mb, 1),
        "macro_f1": round(macro_f1, 4),
        "chemical_f1": round(report.get("B-Chemical", {}).get("f1-score", 0), 4),
        "disease_f1": round(report.get("B-Disease", {}).get("f1-score", 0), 4),
        "latency_mean_ms": round(np.mean(latencies), 2),
        "latency_p50_ms": round(np.percentile(latencies, 50), 2),
        "latency_p95_ms": round(np.percentile(latencies, 95), 2),
        "latencies_raw": latencies,
    }


def generate_visualizations(teacher_metrics, student_metrics):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = ["Teacher\n(Bio_ClinicalBERT)", "Student\n(DistilBERT)"]
    colors = ["#2196F3", "#FF9800"]

    f1_vals = [teacher_metrics["macro_f1"] * 100, student_metrics["macro_f1"] * 100]
    bars = axes[0].bar(labels, f1_vals, color=colors, edgecolor="black", alpha=0.8)
    axes[0].set_ylabel("Macro F1 (%)")
    axes[0].set_title("Entity Detection F1")
    axes[0].set_ylim(min(f1_vals) - 10, max(f1_vals) + 5)
    for bar, val in zip(bars, f1_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontweight="bold")

    sizes = [teacher_metrics["model_size_mb"], student_metrics["model_size_mb"]]
    bars = axes[1].bar(labels, sizes, color=colors, edgecolor="black", alpha=0.8)
    axes[1].set_ylabel("Model Size (MB)")
    axes[1].set_title("Model Size")
    for bar, val in zip(bars, sizes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f"{val:.0f}MB", ha="center", fontweight="bold")

    axes[2].hist(teacher_metrics["latencies_raw"], bins=15, alpha=0.6,
                 label="Teacher", color=colors[0])
    axes[2].hist(student_metrics["latencies_raw"], bins=15, alpha=0.6,
                 label="Student", color=colors[1])
    axes[2].set_xlabel("Latency (ms)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Inference Latency Distribution")
    axes[2].legend()

    plt.suptitle("Knowledge Distillation: Teacher vs Student (Clinical NER)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distillation_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {RESULTS_DIR / 'distillation_summary.png'}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Teacher vs Student evaluation\n")

    results = {}

    print("[1/2] Evaluating teacher (Bio_ClinicalBERT)...")
    if TEACHER_DIR.exists():
        teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_DIR)
        teacher_data = load_test_data(teacher_tokenizer)
        teacher_model = AutoModelForTokenClassification.from_pretrained(TEACHER_DIR)
        results["teacher"] = benchmark_model(teacher_model, teacher_data)
        print(f"  Params: {results['teacher']['parameters']:,}")
        print(f"  F1: {results['teacher']['macro_f1']*100:.2f}%")
        print(f"  Latency: {results['teacher']['latency_mean_ms']:.2f} ms")
        del teacher_model
    else:
        print("  SKIPPED — run train_teacher.py first")

    print("\n[2/2] Evaluating student (DistilBERT)...")
    if STUDENT_DIR.exists():
        student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_DIR)
        student_data = load_test_data(student_tokenizer)
        student_model = AutoModelForTokenClassification.from_pretrained(STUDENT_DIR)
        results["student"] = benchmark_model(student_model, student_data)
        print(f"  Params: {results['student']['parameters']:,}")
        print(f"  F1: {results['student']['macro_f1']*100:.2f}%")
        print(f"  Latency: {results['student']['latency_mean_ms']:.2f} ms")
        del student_model
    else:
        print("  SKIPPED — run distill.py first")

    if "teacher" in results and "student" in results:
        t = results["teacher"]
        s = results["student"]

        f1_retention = s["macro_f1"] / t["macro_f1"] * 100
        param_reduction = (1 - s["parameters"] / t["parameters"]) * 100
        speedup = t["latency_mean_ms"] / s["latency_mean_ms"]

        print(f"\n  Parameter reduction: {param_reduction:.1f}%")
        print(f"  F1 retention:        {f1_retention:.1f}%")
        print(f"  Speedup:             {speedup:.2f}x")
        print(f"  Size reduction:      {t['model_size_mb']:.0f}MB -> {s['model_size_mb']:.0f}MB")

        clean = {}
        for k, v in results.items():
            clean[k] = {key: val for key, val in v.items() if key != "latencies_raw"}
        clean["summary"] = {
            "f1_retention_pct": round(f1_retention, 1),
            "param_reduction_pct": round(param_reduction, 1),
            "speedup": round(speedup, 2),
        }

        with open(RESULTS_DIR / "distillation_report.json", "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\n  Report saved: {RESULTS_DIR / 'distillation_report.json'}")

        generate_visualizations(t, s)


if __name__ == "__main__":
    main()
