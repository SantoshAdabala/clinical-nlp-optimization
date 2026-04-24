import argparse
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tqdm import tqdm

TEACHER_PATH = "../01-distillation/models/teacher"
PUBMED_DATA = "data/pubmed/pubmed_abstracts.parquet"
OUTPUT_DIR = "output/weak_labels_pubmed"
MAX_SEQ_LENGTH = 128
CONFIDENCE_THRESHOLD = 0.85
LABEL_LIST = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]


def load_teacher(model_path):
    print(f"  Loading teacher from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def predict_entities(text, model, tokenizer):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=MAX_SEQ_LENGTH, padding=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().numpy()
        probabilities = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()

    token_labels = []
    for idx, (token, pred_id) in enumerate(zip(tokens, predictions)):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        label = LABEL_LIST[pred_id]
        confidence = float(probabilities[idx][pred_id])
        token_labels.append({
            "token": token, "label": label,
            "confidence": round(confidence, 4), "position": idx,
        })

    entities = []
    current = None
    for tl in token_labels:
        if tl["label"].startswith("B-"):
            if current:
                entities.append(current)
            current = {
                "text": tl["token"].replace("##", ""),
                "label": tl["label"][2:],
                "confidence": tl["confidence"],
            }
        elif tl["label"].startswith("I-") and current:
            current["text"] += tl["token"].replace("##", "")
            current["confidence"] = min(current["confidence"], tl["confidence"])
        else:
            if current:
                entities.append(current)
                current = None
    if current:
        entities.append(current)

    bio_tags = []
    clean_tokens = []
    for tl in token_labels:
        tag_id = LABEL_LIST.index(tl["label"])
        # low confidence predictions get demoted to O
        if tl["label"] != "O" and tl["confidence"] < CONFIDENCE_THRESHOLD:
            tag_id = 0
        bio_tags.append(tag_id)
        clean_tokens.append(tl["token"])

    high_conf = [e for e in entities if e["confidence"] >= CONFIDENCE_THRESHOLD]
    return clean_tokens, bio_tags, high_conf, entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=PUBMED_DATA, help="PubMed parquet file")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit docs (for testing)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    print("Weak labeling: PubMed abstracts\n")

    print(f"  Loading PubMed data from {args.input}...")
    df = pd.read_parquet(args.input)
    if args.max_docs:
        df = df.head(args.max_docs)
    print(f"  Abstracts: {len(df)}")
    print(f"  Avg length: {df['text'].str.len().mean():.0f} chars")

    model, tokenizer = load_teacher(TEACHER_PATH)

    print(f"\nLabeling abstracts with teacher model...")

    pipeline_start = time.time()
    results = []
    total_entities = 0
    high_conf_total = 0
    entity_counts = {"Chemical": 0, "Disease": 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        tokens, tags, high_conf, all_ents = predict_entities(
            row["text"], model, tokenizer
        )

        total_entities += len(all_ents)
        high_conf_total += len(high_conf)

        for e in high_conf:
            entity_counts[e["label"]] = entity_counts.get(e["label"], 0) + 1

        results.append({
            "doc_id": row["pmid"],
            "tokens": tokens,
            "tags": tags,
            "num_entities": len(high_conf),
            "avg_confidence": round(
                np.mean([e["confidence"] for e in high_conf]) if high_conf else 0, 4
            ),
        })

    elapsed = time.time() - pipeline_start

    print(f"\n  Total entities found: {total_entities}")
    print(f"  High-confidence (>={CONFIDENCE_THRESHOLD}): {high_conf_total}")
    print(f"  Filter rate: {high_conf_total/max(total_entities,1)*100:.1f}%")
    print(f"  Chemical entities: {entity_counts.get('Chemical', 0)}")
    print(f"  Disease entities: {entity_counts.get('Disease', 0)}")
    print(f"  Time: {elapsed:.1f}s ({len(df)/elapsed:.1f} docs/sec)")

    print(f"\nSaving training data...")

    train_file = output_path / "weak_labels_pubmed.jsonl"
    with open(train_file, "w") as f:
        for r in results:
            record = {
                "tokens": r["tokens"],
                "tags": r["tags"],
                "doc_id": r["doc_id"],
                "source": "pubmed_weak_label",
                "confidence": r["avg_confidence"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"  Saved {len(results)} records to {train_file}")
    print(f"  File size: {train_file.stat().st_size / (1024*1024):.1f} MB")

    report = {
        "pipeline": "Weak Labeling (Teacher -> PubMed -> Silver Labels)",
        "input": args.input,
        "total_abstracts": len(df),
        "total_entities": total_entities,
        "high_confidence_entities": high_conf_total,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "entity_counts": entity_counts,
        "runtime_seconds": round(elapsed, 1),
        "throughput_docs_per_sec": round(len(df) / elapsed, 1),
        "output_file": str(train_file),
        "teacher_model": TEACHER_PATH,
    }
    with open("results/weak_labeling_pubmed_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Output: {train_file}")
    print(f"  Report: results/weak_labeling_pubmed_report.json")


if __name__ == "__main__":
    main()
