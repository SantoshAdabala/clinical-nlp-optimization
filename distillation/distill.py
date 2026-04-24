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
STUDENT_BASE = "distilbert-base-uncased"
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 128
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
TEMPERATURE = 4.0
ALPHA = 0.5
SEED = 42

MODELS_DIR = Path("models")
STUDENT_DIR = MODELS_DIR / "student"
RESULTS_DIR = Path("results")

LABEL_LIST = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def setup_dirs():
    for d in [MODELS_DIR, STUDENT_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


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

    tokenized = dataset.map(
        tokenize_and_align,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized.set_format("torch")

    return tokenized


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """KL(soft_student || soft_teacher) * T² + CE(student, labels)"""
    soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    mask = (labels != -100).unsqueeze(-1).expand_as(soft_student)
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
    all_preds = []
    all_labels = []

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
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(len(LABEL_LIST))),
        target_names=LABEL_LIST,
        output_dict=True,
        zero_division=0,
    )

    return {
        "macro_f1": round(macro_f1, 4),
        "chemical_f1": round(report.get("B-Chemical", {}).get("f1-score", 0), 4),
        "disease_f1": round(report.get("B-Disease", {}).get("f1-score", 0), 4),
    }


def main():
    setup_dirs()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Distillation: Bio_ClinicalBERT -> DistilBERT")
    print(f"Device: {device}")

    # Load teacher
    print("\n[1/5] Loading teacher model...")
    if not TEACHER_MODEL_PATH.exists():
        print("  ERROR: Teacher model not found. Run train_teacher.py first.")
        return

    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH)
    teacher = AutoModelForTokenClassification.from_pretrained(TEACHER_MODEL_PATH)
    teacher.to(device)
    teacher.eval()

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher parameters: {teacher_params:,}")

    # Init student
    print("\n[2/5] Initializing student model...")
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE)
    student = AutoModelForTokenClassification.from_pretrained(
        STUDENT_BASE,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    student.to(device)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"  Student parameters: {student_params:,}")
    print(f"  Compression: {teacher_params:,} -> {student_params:,} "
          f"({student_params/teacher_params*100:.1f}%)")

    # Need data tokenized for both teacher and student
    print("\n[3/5] Loading and tokenizing data...")
    teacher_data = load_data(teacher_tokenizer)
    student_data = load_data(student_tokenizer)

    train_teacher_loader = DataLoader(
        teacher_data["train"], batch_size=BATCH_SIZE, shuffle=False
    )
    train_student_loader = DataLoader(
        student_data["train"], batch_size=BATCH_SIZE, shuffle=False
    )
    val_student_loader = DataLoader(
        student_data["validation"], batch_size=BATCH_SIZE
    )
    test_student_loader = DataLoader(
        student_data["test"], batch_size=BATCH_SIZE
    )

    print(f"  Train batches: {len(train_student_loader)}")

    # Training
    print(f"\n[4/5] Starting distillation (T={TEMPERATURE}, α={ALPHA})...")

    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_f1 = 0.0
    history = []

    for epoch in range(NUM_EPOCHS):
        student.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            zip(train_teacher_loader, train_student_loader),
            total=len(train_student_loader),
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
        )

        for teacher_batch, student_batch in pbar:
            with torch.no_grad():
                teacher_input_ids = teacher_batch["input_ids"].to(device)
                teacher_attention = teacher_batch["attention_mask"].to(device)
                teacher_outputs = teacher(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention,
                )

            student_input_ids = student_batch["input_ids"].to(device)
            student_attention = student_batch["attention_mask"].to(device)
            labels = student_batch["labels"].to(device)

            student_outputs = student(
                input_ids=student_input_ids,
                attention_mask=student_attention,
            )

            # teacher and student may tokenize differently
            min_seq_len = min(
                teacher_outputs.logits.size(1),
                student_outputs.logits.size(1),
            )
            teacher_logits = teacher_outputs.logits[:, :min_seq_len, :]
            student_logits = student_outputs.logits[:, :min_seq_len, :]
            aligned_labels = labels[:, :min_seq_len]

            loss = distillation_loss(
                student_logits, teacher_logits, aligned_labels,
                temperature=TEMPERATURE, alpha=ALPHA,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / num_batches

        val_metrics = evaluate_ner(student, val_student_loader, device)

        print(f"\n  Epoch {epoch+1}: loss={avg_loss:.4f}, "
              f"val_f1={val_metrics['macro_f1']*100:.2f}%")

        history.append({
            "epoch": epoch + 1,
            "loss": round(avg_loss, 4),
            "val_macro_f1": val_metrics["macro_f1"],
        })

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            student.save_pretrained(STUDENT_DIR)
            student_tokenizer.save_pretrained(STUDENT_DIR)
            print(f"  New best model saved (F1: {best_f1*100:.2f}%)")

    # Final eval
    print("\n[5/5] Final evaluation on test set...")

    student = AutoModelForTokenClassification.from_pretrained(STUDENT_DIR)
    student.to(device)

    test_metrics = evaluate_ner(student, test_student_loader, device)

    print(f"\n  Teacher params: {teacher_params:,}")
    print(f"  Student params: {student_params:,}")
    print(f"  Compression:    {(1 - student_params/teacher_params)*100:.1f}% fewer params")
    print(f"  Student F1:     {test_metrics['macro_f1']*100:.2f}%")
    print(f"  Chemical F1:    {test_metrics['chemical_f1']*100:.2f}%")
    print(f"  Disease F1:     {test_metrics['disease_f1']*100:.2f}%")

    report = {
        "pipeline": "Knowledge Distillation (Clinical NER)",
        "teacher": {
            "model": "Bio_ClinicalBERT",
            "parameters": teacher_params,
        },
        "student": {
            "model": STUDENT_BASE,
            "parameters": student_params,
        },
        "distillation_config": {
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
        "compression_ratio": round(teacher_params / student_params, 2),
        "param_reduction_pct": round((1 - student_params / teacher_params) * 100, 1),
        "student_test_metrics": test_metrics,
        "training_history": history,
    }

    with open(RESULTS_DIR / "distillation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {RESULTS_DIR / 'distillation_report.json'}")

    return report


if __name__ == "__main__":
    main()
