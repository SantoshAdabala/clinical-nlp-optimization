from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
)

# RED method: Rate, Errors, Duration

REQUEST_COUNT = Counter(
    "ner_inference_requests_total",
    "Total number of NER inference requests",
    ["model_version", "status"],
)

REQUEST_LATENCY = Histogram(
    "ner_inference_latency_seconds",
    "Inference latency in seconds",
    ["model_version"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

ERROR_COUNT = Counter(
    "ner_inference_errors_total",
    "Total number of inference errors",
    ["model_version", "error_type"],
)

# ML-specific

ENTITY_COUNT = Counter(
    "ner_entities_detected_total",
    "Total entities detected by type",
    ["entity_type", "model_version"],
)

INPUT_LENGTH = Histogram(
    "ner_input_length_tokens",
    "Input text length in tokens",
    ["model_version"],
    buckets=[16, 32, 64, 128, 256, 512],
)

CONFIDENCE_SCORE = Histogram(
    "ner_prediction_confidence",
    "Model confidence score for predictions",
    ["model_version"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)

BATCH_SIZE = Histogram(
    "ner_batch_size",
    "Batch size per request",
    ["model_version"],
    buckets=[1, 2, 4, 8, 16, 32],
)

# System

MODEL_LOADED = Gauge(
    "ner_model_loaded",
    "Whether the model is loaded and ready",
    ["model_version"],
)

ACTIVE_REQUESTS = Gauge(
    "ner_active_requests",
    "Number of currently processing requests",
)

MODEL_INFO = Info(
    "ner_model",
    "Model metadata",
)

# SLA

SLA_LATENCY_THRESHOLD_MS = 50.0

SLA_VIOLATIONS = Counter(
    "ner_sla_violations_total",
    "Number of requests exceeding SLA latency threshold",
    ["model_version"],
)

REQUESTS_WITHIN_SLA = Counter(
    "ner_requests_within_sla_total",
    "Number of requests within SLA latency threshold",
    ["model_version"],
)
