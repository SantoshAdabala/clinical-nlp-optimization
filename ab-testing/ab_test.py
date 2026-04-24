import argparse
import json
import time
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import f1_score, classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

MODEL_A_PATH = Path("../01-distillation/models/teacher")
MODEL_B_PATH = Path("../01-distillation/models/student_v2")
FALLBACK_B_PATH = Path("../01-distillation/models/student")

MAX_SEQ_LENGTH = 128
SEED = 42
CONFIDENCE_LEVEL = 0.95

LABEL_LIST = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
RESULTS_DIR = Path("results")


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


def run_single_inference(model, input_ids, attention_mask, device="cpu"):
    model.eval()
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.perf_counter()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        latency_ms = (time.perf_counter() - start) * 1000

    preds = torch.argmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
    return preds, latency_ms


def run_ab_experiment(model_a, model_b, dataset, tokenizer_a, tokenizer_b,
                      num_samples=None, device="cpu"):
    print("Running A/B experiment...\n")

    random.seed(SEED)
    np.random.seed(SEED)

    dataset_a = load_test_data(tokenizer_a)
    dataset_b = load_test_data(tokenizer_b)

    if num_samples is None:
        num_samples = len(dataset_a)
    num_samples = min(num_samples, len(dataset_a))

    experiment_log = []

    for i in tqdm(range(num_samples), desc="A/B Testing"):
        sample_a = dataset_a[i]
        sample_b = dataset_b[i]
        labels = sample_a["labels"].numpy()

        assigned_group = random.choice(["A", "B"])

        preds_a, latency_a = run_single_inference(
            model_a, sample_a["input_ids"], sample_a["attention_mask"], device
        )
        preds_b, latency_b = run_single_inference(
            model_b, sample_b["input_ids"], sample_b["attention_mask"], device
        )

        valid_mask = labels != -100
        valid_labels = labels[valid_mask]
        valid_preds_a = preds_a[:len(labels)][valid_mask]
        valid_preds_b = preds_b[:len(labels)][valid_mask]

        correct_a = np.sum(valid_preds_a == valid_labels)
        correct_b = np.sum(valid_preds_b == valid_labels)
        total_tokens = len(valid_labels)

        accuracy_a = correct_a / total_tokens if total_tokens > 0 else 0
        accuracy_b = correct_b / total_tokens if total_tokens > 0 else 0

        f1_a = f1_score(valid_labels, valid_preds_a, average="macro", zero_division=0)
        f1_b = f1_score(valid_labels, valid_preds_b, average="macro", zero_division=0)

        experiment_log.append({
            "sample_id": i,
            "assigned_group": assigned_group,
            "latency_a_ms": round(latency_a, 3),
            "latency_b_ms": round(latency_b, 3),
            "accuracy_a": round(accuracy_a, 4),
            "accuracy_b": round(accuracy_b, 4),
            "f1_a": round(f1_a, 4),
            "f1_b": round(f1_b, 4),
            "num_tokens": total_tokens,
            "correct_a": int(correct_a),
            "correct_b": int(correct_b),
        })

    df = pd.DataFrame(experiment_log)
    print(f"  Samples tested: {num_samples}")
    print(f"  Avg latency A: {df['latency_a_ms'].mean():.2f} ms")
    print(f"  Avg latency B: {df['latency_b_ms'].mean():.2f} ms")
    print(f"  Avg F1 A: {df['f1_a'].mean()*100:.2f}%")
    print(f"  Avg F1 B: {df['f1_b'].mean()*100:.2f}%")

    return df


def run_statistical_tests(df):
    print("\nStatistical analysis:\n")

    results = {}

    latency_a = df["latency_a_ms"].values
    latency_b = df["latency_b_ms"].values

    u_stat, u_pvalue = stats.mannwhitneyu(latency_a, latency_b, alternative="two-sided")

    latency_diff = latency_a - latency_b
    mean_diff = np.mean(latency_diff)
    ci_low, ci_high = stats.t.interval(
        CONFIDENCE_LEVEL,
        df=len(latency_diff) - 1,
        loc=mean_diff,
        scale=stats.sem(latency_diff),
    )

    results["latency"] = {
        "test": "Mann-Whitney U",
        "mean_a_ms": round(np.mean(latency_a), 2),
        "mean_b_ms": round(np.mean(latency_b), 2),
        "mean_difference_ms": round(mean_diff, 2),
        "ci_95_low": round(ci_low, 2),
        "ci_95_high": round(ci_high, 2),
        "u_statistic": round(u_stat, 2),
        "p_value": round(u_pvalue, 6),
        "significant": u_pvalue < (1 - CONFIDENCE_LEVEL),
        "winner": "A" if mean_diff > 0 else "B" if mean_diff < 0 else "tie",
    }

    print(f"  Latency (Mann-Whitney U)")
    print(f"    A: {results['latency']['mean_a_ms']:.2f} ms, B: {results['latency']['mean_b_ms']:.2f} ms")
    print(f"    Diff: {mean_diff:.2f} ms (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")
    print(f"    p={u_pvalue:.6f}, significant: {'yes' if results['latency']['significant'] else 'no'}")

    f1_a = df["f1_a"].values
    f1_b = df["f1_b"].values

    f1_diff = f1_a - f1_b
    nonzero_diff = f1_diff[f1_diff != 0]

    if len(nonzero_diff) > 10:
        w_stat, w_pvalue = stats.wilcoxon(nonzero_diff)
    else:
        w_stat, w_pvalue = 0, 1.0

    f1_mean_diff = np.mean(f1_diff)
    if len(f1_diff) > 1:
        f1_ci_low, f1_ci_high = stats.t.interval(
            CONFIDENCE_LEVEL,
            df=len(f1_diff) - 1,
            loc=f1_mean_diff,
            scale=stats.sem(f1_diff),
        )
    else:
        f1_ci_low, f1_ci_high = f1_mean_diff, f1_mean_diff

    results["f1_score"] = {
        "test": "Wilcoxon signed-rank",
        "mean_a": round(np.mean(f1_a), 4),
        "mean_b": round(np.mean(f1_b), 4),
        "mean_difference": round(f1_mean_diff, 4),
        "ci_95_low": round(f1_ci_low, 4),
        "ci_95_high": round(f1_ci_high, 4),
        "w_statistic": round(float(w_stat), 2),
        "p_value": round(float(w_pvalue), 6),
        "significant": float(w_pvalue) < (1 - CONFIDENCE_LEVEL),
        "winner": "A" if f1_mean_diff > 0.001 else "B" if f1_mean_diff < -0.001 else "tie",
    }

    print(f"\n  F1 (Wilcoxon signed-rank)")
    print(f"    A: {results['f1_score']['mean_a']*100:.2f}%, B: {results['f1_score']['mean_b']*100:.2f}%")
    print(f"    Diff: {f1_mean_diff*100:.2f}%")
    print(f"    p={float(w_pvalue):.6f}, significant: {'yes' if results['f1_score']['significant'] else 'no'}")

    # the student trades F1 for speed/size — is the trade-off worth it?
    f1_a_val = results["f1_score"]["mean_a"]
    f1_b_val = results["f1_score"]["mean_b"]
    f1_retention = f1_b_val / f1_a_val * 100 if f1_a_val > 0 else 0

    if f1_retention >= 90:
        recommendation = f"DEPLOY Student — retains {f1_retention:.1f}% of teacher F1 with {results['latency']['mean_difference_ms']:.1f}ms faster inference"
    elif f1_retention >= 80:
        recommendation = f"Student retains {f1_retention:.1f}% of teacher F1 — acceptable for most use cases"
    elif f1_retention >= 70:
        recommendation = f"Student retains {f1_retention:.1f}% of teacher F1 — consider for latency-critical deployments"
    else:
        recommendation = f"Student retains only {f1_retention:.1f}% of teacher F1 — further tuning recommended"

    results["recommendation"] = recommendation
    results["f1_retention_pct"] = round(f1_retention, 1)
    results["confidence_level"] = CONFIDENCE_LEVEL
    results["num_samples"] = len(df)

    print(f"\n  Recommendation: {recommendation}")

    return results


def generate_visualizations(df, test_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(df["latency_a_ms"], bins=30, alpha=0.6, label="Model A (Teacher)", color="#2196F3")
    axes[0, 0].hist(df["latency_b_ms"], bins=30, alpha=0.6, label="Model B (Optimized)", color="#FF9800")
    axes[0, 0].axvline(df["latency_a_ms"].mean(), color="#2196F3", linestyle="--", linewidth=2)
    axes[0, 0].axvline(df["latency_b_ms"].mean(), color="#FF9800", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("Latency (ms)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Latency Distribution")
    axes[0, 0].legend()

    axes[0, 1].hist(df["f1_a"], bins=30, alpha=0.6, label="Model A", color="#2196F3")
    axes[0, 1].hist(df["f1_b"], bins=30, alpha=0.6, label="Model B", color="#FF9800")
    axes[0, 1].set_xlabel("Per-Sample F1")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("F1 Score Distribution")
    axes[0, 1].legend()

    axes[1, 0].plot(df["sample_id"], df["latency_a_ms"], alpha=0.3, label="Model A", color="#2196F3")
    axes[1, 0].plot(df["sample_id"], df["latency_b_ms"], alpha=0.3, label="Model B", color="#FF9800")
    window = max(len(df) // 20, 5)
    axes[1, 0].plot(df["sample_id"], df["latency_a_ms"].rolling(window).mean(),
                    label="A (rolling avg)", color="#1565C0", linewidth=2)
    axes[1, 0].plot(df["sample_id"], df["latency_b_ms"].rolling(window).mean(),
                    label="B (rolling avg)", color="#E65100", linewidth=2)
    axes[1, 0].set_xlabel("Sample #")
    axes[1, 0].set_ylabel("Latency (ms)")
    axes[1, 0].set_title("Latency Over Time")
    axes[1, 0].legend(fontsize=8)

    metrics = ["Latency (ms)", "F1 Score"]
    a_vals = [df["latency_a_ms"].mean(), df["f1_a"].mean() * 100]
    b_vals = [df["latency_b_ms"].mean(), df["f1_b"].mean() * 100]

    x = np.arange(len(metrics))
    width = 0.35
    bars_a = axes[1, 1].bar(x - width/2, a_vals, width, label="Model A (Teacher)", color="#2196F3", alpha=0.8)
    bars_b = axes[1, 1].bar(x + width/2, b_vals, width, label="Model B (Optimized)", color="#FF9800", alpha=0.8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_title("Summary Comparison")
    axes[1, 1].legend()

    for bar, val in zip(bars_a, a_vals):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", fontsize=9)
    for bar, val in zip(bars_b, b_vals):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", fontsize=9)

    rec = test_results.get("recommendation", "")
    plt.suptitle(f"A/B Test: Teacher vs Optimized (Clinical NER)\n{rec}",
                 fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ab_test_summary_teacher_vs_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Visualization saved: {RESULTS_DIR / 'ab_test_summary.png'}")


def main():
    parser = argparse.ArgumentParser(description="A/B Testing: Teacher vs Optimized Model")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of test samples (default: 200)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    print("A/B test: Teacher (A) vs Optimized (B)\n")

    print("  Loading Model A (Teacher)...")
    if not MODEL_A_PATH.exists():
        print("  ERROR: Teacher model not found. Run Component 1 first.")
        return
    tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A_PATH)
    model_a = AutoModelForTokenClassification.from_pretrained(MODEL_A_PATH)
    print(f"  Model A: {MODEL_A_PATH}")
    print(f"  Params: {sum(p.numel() for p in model_a.parameters()):,}")

    print("\n  Loading Model B (Optimized)...")
    if MODEL_B_PATH.exists():
        model_b_path = MODEL_B_PATH
    elif FALLBACK_B_PATH.exists():
        model_b_path = FALLBACK_B_PATH
        print("  (Using distilled student as fallback)")
    else:
        print("  ERROR: No Model B found. Run Component 1 or 3 first.")
        return

    tokenizer_b = AutoTokenizer.from_pretrained(model_b_path)
    model_b = AutoModelForTokenClassification.from_pretrained(model_b_path)
    print(f"  Model B: {model_b_path}")
    print(f"  Params: {sum(p.numel() for p in model_b.parameters()):,}")

    df = run_ab_experiment(
        model_a, model_b,
        None,
        tokenizer_a, tokenizer_b,
        num_samples=args.num_samples,
    )

    test_results = run_statistical_tests(df)

    df.to_csv(RESULTS_DIR / "experiment_log_teacher_vs_v2.csv", index=False)
    print(f"\n  Experiment log: {RESULTS_DIR / 'experiment_log_teacher_vs_v2.csv'}")

    with open(RESULTS_DIR / "ab_test_report_teacher_vs_v2.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"  Test report: {RESULTS_DIR / 'ab_test_report_teacher_vs_v2.json'}")

    generate_visualizations(df, test_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
