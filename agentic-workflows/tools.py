import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

DISTILLATION_REPORT = Path("../01-distillation/results/distillation_report.json")
OPTIMIZATION_REPORT = Path("../03-model-optimization/results/benchmark_report.json")
PIPELINE_STATS = Path("../02-distributed-training/results/pipeline_stats.json")
RESULTS_DIR = Path("results")

THRESHOLDS = {
    "max_f1_degradation_pct": 2.0,
    "min_speedup_factor": 2.0,
    "max_latency_ms": 50.0,
    "min_f1_retention_pct": 85.0,
    "min_sparsity_pct": 35.0,
}


def read_benchmark_report(report_path: str = "") -> str:
    if not report_path:
        path = OPTIMIZATION_REPORT
    else:
        path = Path(report_path)

    if not path.exists():
        return json.dumps({"error": f"Report not found: {path}"})

    with open(path) as f:
        data = json.load(f)

    return json.dumps(data, indent=2)


def read_all_reports() -> str:
    reports = {}

    if DISTILLATION_REPORT.exists():
        with open(DISTILLATION_REPORT) as f:
            reports["distillation"] = json.load(f)

    if OPTIMIZATION_REPORT.exists():
        with open(OPTIMIZATION_REPORT) as f:
            reports["optimization"] = json.load(f)

    if PIPELINE_STATS.exists():
        with open(PIPELINE_STATS) as f:
            reports["distributed_pipeline"] = json.load(f)

    if not reports:
        return json.dumps({"error": "No reports found. Run Components 1-3 first."})

    return json.dumps(reports, indent=2)


def analyze_regressions() -> str:
    findings = []
    status = "PASS"

    if OPTIMIZATION_REPORT.exists():
        with open(OPTIMIZATION_REPORT) as f:
            opt = json.load(f)

        summary = opt.get("summary", {})

        f1_drop = summary.get("f1_degradation_pct", 0)
        if f1_drop > THRESHOLDS["max_f1_degradation_pct"]:
            findings.append({
                "component": "Model Optimization",
                "metric": "F1 Degradation",
                "value": f"{f1_drop:.2f}%",
                "threshold": f"{THRESHOLDS['max_f1_degradation_pct']}%",
                "severity": "HIGH",
                "message": f"F1 dropped {f1_drop:.2f}% after optimization (threshold: {THRESHOLDS['max_f1_degradation_pct']}%)",
            })
            status = "FAIL"
        else:
            findings.append({
                "component": "Model Optimization",
                "metric": "F1 Degradation",
                "value": f"{f1_drop:.2f}%",
                "threshold": f"{THRESHOLDS['max_f1_degradation_pct']}%",
                "severity": "OK",
                "message": f"F1 degradation within range ({f1_drop:.2f}% < {THRESHOLDS['max_f1_degradation_pct']}%)",
            })

        speedup = summary.get("speedup_factor", 0)
        if speedup < THRESHOLDS["min_speedup_factor"]:
            findings.append({
                "component": "Model Optimization",
                "metric": "Speedup Factor",
                "value": f"{speedup:.2f}x",
                "threshold": f"{THRESHOLDS['min_speedup_factor']}x",
                "severity": "MEDIUM",
                "message": f"Speedup {speedup:.2f}x below target {THRESHOLDS['min_speedup_factor']}x (may be hardware-dependent — INT8 optimized for x86 AVX-512)",
            })
            if status != "FAIL":
                status = "WARN"
        else:
            findings.append({
                "component": "Model Optimization",
                "metric": "Speedup Factor",
                "value": f"{speedup:.2f}x",
                "threshold": f"{THRESHOLDS['min_speedup_factor']}x",
                "severity": "OK",
                "message": f"Speedup meets target ({speedup:.2f}x >= {THRESHOLDS['min_speedup_factor']}x)",
            })

        sparsity = summary.get("sparsity_pct", 0)
        if sparsity < THRESHOLDS["min_sparsity_pct"]:
            findings.append({
                "component": "Model Optimization",
                "metric": "Sparsity",
                "value": f"{sparsity:.1f}%",
                "threshold": f"{THRESHOLDS['min_sparsity_pct']}%",
                "severity": "MEDIUM",
                "message": f"Sparsity {sparsity:.1f}% below target {THRESHOLDS['min_sparsity_pct']}%",
            })
        else:
            findings.append({
                "component": "Model Optimization",
                "metric": "Sparsity",
                "value": f"{sparsity:.1f}%",
                "threshold": f"{THRESHOLDS['min_sparsity_pct']}%",
                "severity": "OK",
                "message": f"Sparsity target met ({sparsity:.1f}% >= {THRESHOLDS['min_sparsity_pct']}%)",
            })

    if DISTILLATION_REPORT.exists():
        with open(DISTILLATION_REPORT) as f:
            dist = json.load(f)

        dist_summary = dist.get("summary", {})
        retention = dist_summary.get("f1_retention_pct", 0)

        if retention < THRESHOLDS["min_f1_retention_pct"]:
            findings.append({
                "component": "Distillation",
                "metric": "F1 Retention",
                "value": f"{retention:.1f}%",
                "threshold": f"{THRESHOLDS['min_f1_retention_pct']}%",
                "severity": "MEDIUM",
                "message": f"Student retains {retention:.1f}% of teacher F1 (target: {THRESHOLDS['min_f1_retention_pct']}%)",
            })
            if status == "PASS":
                status = "WARN"
        else:
            findings.append({
                "component": "Distillation",
                "metric": "F1 Retention",
                "value": f"{retention:.1f}%",
                "threshold": f"{THRESHOLDS['min_f1_retention_pct']}%",
                "severity": "OK",
                "message": f"F1 retention meets target ({retention:.1f}% >= {THRESHOLDS['min_f1_retention_pct']}%)",
            })

    result = {
        "overall_status": status,
        "findings": findings,
        "thresholds": THRESHOLDS,
        "timestamp": datetime.now().isoformat(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "regression_analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    return json.dumps(result, indent=2)


def generate_evaluation_summary() -> str:
    sections = []
    sections.append("# ML Pipeline Evaluation Summary")
    sections.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    if DISTILLATION_REPORT.exists():
        with open(DISTILLATION_REPORT) as f:
            dist = json.load(f)

        teacher = dist.get("teacher", {})
        student = dist.get("student", {})
        summary = dist.get("summary", {})

        sections.append("## Component 1: Knowledge Distillation\n")
        sections.append("| Metric | Teacher | Student |")
        sections.append("|--------|---------|---------|")

        t_metrics = teacher if "macro_f1" in teacher else {}
        s_metrics = student if "macro_f1" in student else {}

        if t_metrics and s_metrics:
            sections.append(f"| Parameters | {teacher.get('parameters', 'N/A'):,} | {student.get('parameters', 'N/A'):,} |")
            sections.append(f"| Macro F1 | {t_metrics.get('macro_f1', 0)*100:.2f}% | {s_metrics.get('macro_f1', 0)*100:.2f}% |")
            sections.append(f"| Latency | {t_metrics.get('latency_mean_ms', 'N/A')} ms | {s_metrics.get('latency_mean_ms', 'N/A')} ms |")
            sections.append(f"| Size | {t_metrics.get('model_size_mb', 'N/A')} MB | {s_metrics.get('model_size_mb', 'N/A')} MB |")

        sections.append(f"\n**F1 Retention:** {summary.get('f1_retention_pct', 'N/A')}%")
        sections.append(f"**Speedup:** {summary.get('speedup', 'N/A')}x")
        sections.append(f"**Parameter Reduction:** {summary.get('param_reduction_pct', 'N/A')}%\n")

    if OPTIMIZATION_REPORT.exists():
        with open(OPTIMIZATION_REPORT) as f:
            opt = json.load(f)

        results = opt.get("results", {})
        summary = opt.get("summary", {})
        sizes = opt.get("model_sizes", {})

        sections.append("## Component 3: Model Optimization (Pruning + INT8)\n")
        sections.append("| Variant | Macro F1 | Latency |")
        sections.append("|---------|----------|---------|")

        for variant_name, metrics in results.items():
            f1 = metrics.get("macro_f1", 0)
            lat = metrics.get("avg_latency_ms", "N/A")
            display_name = variant_name.replace("_", " ").title()
            sections.append(f"| {display_name} | {f1*100:.2f}% | {lat} ms |")

        sections.append(f"\n**Speedup:** {summary.get('speedup_factor', 'N/A')}x")
        sections.append(f"**F1 Degradation:** {summary.get('f1_degradation_pct', 'N/A')}%")
        sections.append(f"**Sparsity:** {summary.get('sparsity_pct', 'N/A')}%")
        sections.append(f"**Model Size:** {sizes.get('onnx_size_mb', 'N/A')} MB -> {sizes.get('quantized_size_mb', 'N/A')} MB\n")

        platform = opt.get("platform_note", {})
        if platform:
            sections.append(f"**Note:** {platform.get('latency_caveat', '')}\n")

    if PIPELINE_STATS.exists():
        with open(PIPELINE_STATS) as f:
            pipe = json.load(f)

        sections.append("## Component 2: Distributed Pipeline\n")
        sections.append(f"- **Total Documents:** {pipe.get('total_documents', 'N/A'):,}")
        sections.append(f"- **Total Words:** {pipe.get('total_words', 'N/A'):,}")
        sections.append(f"- **Avg Document Length:** {pipe.get('avg_document_length', 'N/A')} words")

        metrics = pipe.get("pipeline_metrics", {})
        if metrics:
            sections.append(f"- **Mode:** {metrics.get('mode', 'N/A')}")
            sections.append(f"- **Runtime:** {metrics.get('total_runtime_seconds', 'N/A')}s")
        sections.append("")

    if (RESULTS_DIR / "regression_analysis.json").exists():
        with open(RESULTS_DIR / "regression_analysis.json") as f:
            reg = json.load(f)

        sections.append("## Regression Analysis\n")
        sections.append(f"**Overall Status:** {reg.get('overall_status', 'UNKNOWN')}\n")

        for finding in reg.get("findings", []):
            icon = "✅" if finding["severity"] == "OK" else "⚠️" if finding["severity"] == "MEDIUM" else "❌"
            sections.append(f"- {icon} **{finding['metric']}** ({finding['component']}): {finding['message']}")

        sections.append("")

    summary_text = "\n".join(sections)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "evaluation_summary.md", "w") as f:
        f.write(summary_text)

    return summary_text


def check_alert_conditions() -> str:
    if not (RESULTS_DIR / "regression_analysis.json").exists():
        analyze_regressions()

    with open(RESULTS_DIR / "regression_analysis.json") as f:
        reg = json.load(f)

    alerts = []
    for finding in reg.get("findings", []):
        if finding["severity"] in ("HIGH", "MEDIUM"):
            alerts.append({
                "channel": "#ml-alerts",
                "severity": finding["severity"],
                "component": finding["component"],
                "metric": finding["metric"],
                "message": finding["message"],
                "value": finding["value"],
                "threshold": finding["threshold"],
            })

    result = {
        "alerts_triggered": len(alerts),
        "overall_status": reg.get("overall_status", "UNKNOWN"),
        "alerts": alerts,
        "action_required": len(alerts) > 0,
        "notification_targets": {
            "slack": "#ml-alerts",
            "email": "[ml-team-email]",
        },
        "note": "In production, these would be sent via Slack webhook or SES email.",
    }

    return json.dumps(result, indent=2)


def analyze_clinical_document(text: str) -> str:
    """Run NER on clinical text, return detected entities."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model_path = Path("../01-distillation/models/teacher")
    if not model_path.exists():
        return json.dumps({"error": "Teacher model not found. Run Component 1 first."})

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    label_list = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().numpy()
        probabilities = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()

    entities = []
    current_entity = None

    for idx, (token, pred_id) in enumerate(zip(tokens, predictions)):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        label = label_list[pred_id]
        confidence = float(probabilities[idx][pred_id])

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": token.replace("##", ""),
                "label": label[2:],
                "confidence": round(confidence, 4),
                "start_token": idx,
            }
        elif label.startswith("I-") and current_entity:
            current_entity["text"] += token.replace("##", "")
            current_entity["confidence"] = round(
                min(current_entity["confidence"], confidence), 4
            )
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    result = {
        "input_text": text,
        "num_tokens": len(tokens),
        "entities_found": len(entities),
        "entities": entities,
        "entity_summary": {},
    }

    for e in entities:
        etype = e["label"]
        result["entity_summary"][etype] = result["entity_summary"].get(etype, 0) + 1

    return json.dumps(result, indent=2)


def analyze_document_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})

    text = path.read_text().strip()
    if not text:
        return json.dumps({"error": f"File is empty: {file_path}"})

    return analyze_clinical_document(text)
