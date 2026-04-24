import json
import torch
import numpy as np
from pathlib import Path
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
import warnings

warnings.filterwarnings("ignore")

TEACHER_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 128
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42

MODELS_DIR = Path("models")
TEACHER_DIR = MODELS_DIR / "teacher"
RESULTS_DIR = Path("results")

LABEL_LIST = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def setup_dirs():
    for d in [MODELS_DIR, TEACHER_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data(tokenizer):
    """Load BC5CDR and tokenize with label alignment."""
    dataset = load_dataset("tner/bc5cdr")

    def tokenize_and_align_labels(examples):
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
                    # B- → I- for subword continuations
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

    tokenized = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print(f"  Train: {len(tokenized['train'])} samples")
    print(f"  Val:   {len(tokenized['validation'])} samples")
    print(f"  Test:  {len(tokenized['test'])} samples")

    return tokenized


def evaluate_model(model, dataset, tokenizer, device="cpu", split="test"):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    eval_data = dataset[split]
    eval_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dataloader = torch.utils.data.DataLoader(eval_data, batch_size=BATCH_SIZE)

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
        "report": report,
    }


def main():
    setup_dirs()
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

    if TEACHER_DIR.exists() and (TEACHER_DIR / "config.json").exists():
        print("Teacher model already exists. Loading...")
        model = AutoModelForTokenClassification.from_pretrained(TEACHER_DIR)
        dataset = load_and_prepare_data(tokenizer)

        print("\nEvaluating existing teacher...")
        metrics = evaluate_model(model, dataset, tokenizer)
        print(f"  Macro F1: {metrics['macro_f1']*100:.2f}%")
        print(f"  Chemical F1: {metrics['chemical_f1']*100:.2f}%")
        print(f"  Disease F1: {metrics['disease_f1']*100:.2f}%")
        return model, metrics

    dataset = load_and_prepare_data(tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        TEACHER_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Training {TEACHER_MODEL}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Epochs: {NUM_EPOCHS}")

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "teacher_checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=SEED,
        report_to="none",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(TEACHER_DIR)
    tokenizer.save_pretrained(TEACHER_DIR)
    print(f"\n  Teacher saved to: {TEACHER_DIR}")

    metrics = evaluate_model(model, dataset, tokenizer)
    print(f"\n  Macro F1: {metrics['macro_f1']*100:.2f}%")
    print(f"  Chemical F1: {metrics['chemical_f1']*100:.2f}%")
    print(f"  Disease F1: {metrics['disease_f1']*100:.2f}%")

    teacher_report = {
        "model": TEACHER_MODEL,
        "parameters": num_params,
        "epochs": NUM_EPOCHS,
        "metrics": {k: v for k, v in metrics.items() if k != "report"},
    }
    with open(RESULTS_DIR / "teacher_metrics.json", "w") as f:
        json.dump(teacher_report, f, indent=2)

    return model, metrics


if __name__ == "__main__":
    main()
