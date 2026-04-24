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
from optimum.onnxruntime import ORTModelForTokenClassification
from sklearn.metrics import classification_report, f1_score
import warnings

warnings.filterwarnings("ignore")

BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
FINETUNED_DIR = Path("models/finetuned_ner")
PRUNED_DIR = Path("models/pruned_model")
QUANTIZED_DIR = Path("models/quantized_model")
RESULTS_DIR = Path("results")
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
NUM_WARMUP = 5
NUM_BENCHMARK_RUNS = 50

LABEL_LIST = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]


def load_eval_data(tokenizer):
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


def benchmark_pytorch_ner(model, dataset, device="cpu"):
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
        for _ in range(NUM_BENCHMARK_RUNS):
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
        "latency_mean_ms": round(np.mean(latencies), 2),
        "latency_p50_ms": round(np.percentile(latencies, 50), 2),
        "latency_p95_ms": round(np.percentile(latencies, 95), 2),
        "latency_p99_ms": round(np.percentile(latencies, 99), 2),
        "latency_std_ms": round(np.std(latencies), 2),
        "latencies_raw": latencies,
    }


def benchmark_onnx_ner(model_dir, dataset, tokenizer):
    ort_model = ORTModelForTokenClassification.from_pretrained(model_dir)

    sample = dataset[0]
    inputs = {
        "input_ids": sample["input_ids"].unsqueeze(0),
        "attention_mask": sample["attention_mask"].unsqueeze(0),
    }
    for _ in range(NUM_WARMUP):
        ort_model(**inputs)

    latencies = []
    for _ in range(NUM_BENCHMARK_RUNS):
        start = time.perf_counter()
        ort_model(**inputs)
        latencies.append((time.perf_counter() - start) * 1000)

    all_preds, all_labels = [], []
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i : i + BATCH_SIZE]
        batch_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        outputs = ort_model(**batch_inputs)
        preds = torch.argmax(torch.tensor(outputs.logits), dim=-1).numpy()

        labels = batch["labels"].numpy()
        for pred_seq, label_seq in zip(preds, labels):
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
        "latency_mean_ms": round(np.mean(latencies), 2),
        "latency_p50_ms": round(np.percentile(latencies, 50), 2),
        "latency_p95_ms": round(np.percentile(latencies, 95), 2),
        "latency_p99_ms": round(np.percentile(latencies, 99), 2),
        "latency_std_ms": round(np.std(latencies), 2),
        "latencies_raw": latencies,
    }


def generate_visualizations(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    variants = list(results.keys())
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    f1_scores = [results[v]["macro_f1"] * 100 for v in variants]
    bars = axes[0, 0].bar(variants, f1_scores, color=colors, edgecolor="black", alpha=0.8)
    axes[0, 0].set_ylabel("Macro F1 (%)")
    axes[0, 0].set_title("Entity Detection F1 Score")
    axes[0, 0].set_ylim(min(f1_scores) - 5, max(f1_scores) + 3)
    for bar, f1 in zip(bars, f1_scores):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{f1:.1f}%",
            ha="center",
            fontweight="bold",
        )

    x = np.arange(len(variants))
    width = 0.35
    chem_f1 = [results[v]["chemical_f1"] * 100 for v in variants]
    dis_f1 = [results[v]["disease_f1"] * 100 for v in variants]

    axes[0, 1].bar(x - width / 2, chem_f1, width, label="Chemical (Drug)", color="#E91E63", alpha=0.8)
    axes[0, 1].bar(x + width / 2, dis_f1, width, label="Disease", color="#9C27B0", alpha=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(variants)
    axes[0, 1].set_ylabel("F1 (%)")
    axes[0, 1].set_title("Per-Entity Type F1")
    axes[0, 1].legend()

    means = [results[v]["latency_mean_ms"] for v in variants]
    p95s = [results[v]["latency_p95_ms"] for v in variants]

    bars1 = axes[1, 0].bar(x - width / 2, means, width, label="Mean", color=colors, alpha=0.8)
    bars2 = axes[1, 0].bar(x + width / 2, p95s, width, label="P95", color=colors, alpha=0.5)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(variants)
    axes[1, 0].set_ylabel("Latency (ms)")
    axes[1, 0].set_title("Inference Latency")
    axes[1, 0].axhline(y=20, color="red", linestyle="--", alpha=0.7, label="20ms SLA")
    axes[1, 0].legend()

    for i, variant in enumerate(variants):
        axes[1, 1].hist(
            results[variant]["latencies_raw"],
            bins=20,
            alpha=0.6,
            label=variant,
            color=colors[i],
        )
    axes[1, 1].axvline(x=20, color="red", linestyle="--", alpha=0.7, label="20ms SLA")
    axes[1, 1].set_xlabel("Latency (ms)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Latency Distribution")
    axes[1, 1].legend()

    plt.suptitle(
        "Clinical NER Model Optimization: PHI Detection for Edge Deployment",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "optimization_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved: {RESULTS_DIR / 'optimization_summary.png'}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Clinical NER benchmark: baseline vs pruned vs quantized\n")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    dataset = load_eval_data(tokenizer)
    print(f"  Test samples: {len(dataset)}")

    results = {}

    print("\n[1/3] Benchmarking baseline (FP32)...")
    if FINETUNED_DIR.exists():
        baseline_model = AutoModelForTokenClassification.from_pretrained(FINETUNED_DIR)
        results["Baseline (FP32)"] = benchmark_pytorch_ner(baseline_model, dataset)
        print(f"  Macro F1: {results['Baseline (FP32)']['macro_f1']*100:.2f}%")
        print(f"  Latency: {results['Baseline (FP32)']['latency_mean_ms']:.2f} ms")
        del baseline_model
    else:
        print("  SKIPPED — run optimize.py first.")

    print("\n[2/3] Benchmarking pruned model...")
    if PRUNED_DIR.exists():
        pruned_model = AutoModelForTokenClassification.from_pretrained(PRUNED_DIR)
        results["Pruned (40%)"] = benchmark_pytorch_ner(pruned_model, dataset)
        print(f"  Macro F1: {results['Pruned (40%)']['macro_f1']*100:.2f}%")
        print(f"  Latency: {results['Pruned (40%)']['latency_mean_ms']:.2f} ms")
        del pruned_model
    else:
        print("  SKIPPED — run optimize.py first.")

    print("\n[3/3] Benchmarking quantized model (INT8)...")
    if QUANTIZED_DIR.exists():
        results["Quantized (INT8)"] = benchmark_onnx_ner(
            QUANTIZED_DIR, dataset, tokenizer
        )
        print(f"  Macro F1: {results['Quantized (INT8)']['macro_f1']*100:.2f}%")
        print(f"  Latency: {results['Quantized (INT8)']['latency_mean_ms']:.2f} ms")
    else:
        print("  SKIPPED — run optimize.py first.")

    summary_data = []
    for variant, metrics in results.items():
        row = {
            "variant": variant,
            "macro_f1": metrics["macro_f1"],
            "chemical_f1": metrics["chemical_f1"],
            "disease_f1": metrics["disease_f1"],
            "latency_mean_ms": metrics["latency_mean_ms"],
            "latency_p95_ms": metrics["latency_p95_ms"],
        }
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    print(f"\n{df.to_string(index=False)}")

    if "Baseline (FP32)" in results and "Quantized (INT8)" in results:
        speedup = (
            results["Baseline (FP32)"]["latency_mean_ms"]
            / results["Quantized (INT8)"]["latency_mean_ms"]
        )
        f1_drop = (
            results["Baseline (FP32)"]["macro_f1"]
            - results["Quantized (INT8)"]["macro_f1"]
        )
        print(f"\n  Speedup: {speedup:.2f}x")
        print(f"  F1 degradation: {f1_drop*100:.2f}%")
        meets_sla = results["Quantized (INT8)"]["latency_p95_ms"] < 20
        print(f"  Meets 20ms SLA (P95): {'YES' if meets_sla else 'NO'}")

    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {key: val for key, val in v.items() if key != "latencies_raw"}

    with open(RESULTS_DIR / "benchmark_report.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\n  Report saved: {RESULTS_DIR / 'benchmark_report.json'}")

    if len(results) >= 2:
        print("  Generating visualizations...")
        generate_visualizations(results)


if __name__ == "__main__":
    main()
