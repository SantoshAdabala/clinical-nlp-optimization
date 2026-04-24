import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in ("request_id", "trace_id", "model_version", "latency_ms",
                       "entity_counts", "input_length", "status_code", "error_type"):
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)

        return json.dumps(log_entry)


def get_logger(name="ner_inference"):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(JSONFormatter())
        logger.addHandler(console)

        file_handler = logging.FileHandler(LOG_DIR / "inference.log")
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger


def log_inference(logger, request_id, model_version, latency_ms,
                  input_length, entity_counts, trace_id=None):
    extra = {
        "request_id": request_id,
        "model_version": model_version,
        "latency_ms": round(latency_ms, 2),
        "input_length": input_length,
        "entity_counts": entity_counts,
    }
    if trace_id:
        extra["trace_id"] = trace_id

    logger.info("Inference completed", extra=extra)


def log_error(logger, request_id, model_version, error_type, error_msg, trace_id=None):
    extra = {
        "request_id": request_id,
        "model_version": model_version,
        "error_type": error_type,
    }
    if trace_id:
        extra["trace_id"] = trace_id

    logger.error(f"Inference error: {error_msg}", extra=extra)


def generate_request_id():
    return str(uuid.uuid4())[:8]
