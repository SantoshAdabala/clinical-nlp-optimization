import time
import torch
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from transformers import AutoModelForTokenClassification, AutoTokenizer

from metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT,
    ENTITY_COUNT, INPUT_LENGTH, CONFIDENCE_SCORE,
    MODEL_LOADED, ACTIVE_REQUESTS, MODEL_INFO,
    SLA_VIOLATIONS, REQUESTS_WITHIN_SLA, SLA_LATENCY_THRESHOLD_MS,
)
from logger import get_logger, log_inference, log_error, generate_request_id
from tracing import setup_tracing

import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = Path("../distillation/models/student_v2")
MODEL_VERSION = "v1.0-distilled-student-v2"
MAX_SEQ_LENGTH = 128
LABEL_LIST = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]

model = None
tokenizer = None
teacher_model = None
teacher_tokenizer = None
logger = None
tracer = None

TEACHER_PATH = Path("../distillation/models/teacher")
TEACHER_VERSION = "v1.0-teacher"


class NERRequest(BaseModel):
    text: str
    request_id: Optional[str] = None

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float

class NERResponse(BaseModel):
    request_id: str
    entities: List[Entity]
    latency_ms: float
    model_version: str
    num_tokens: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, teacher_model, teacher_tokenizer, logger, tracer

    logger = get_logger()
    tracer = setup_tracing(export_to_console=False)

    logger.info(f"Loading student model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    logger.info(f"Loading teacher model from {TEACHER_PATH}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH)
    teacher_model = AutoModelForTokenClassification.from_pretrained(TEACHER_PATH)
    teacher_model.eval()

    MODEL_LOADED.labels(model_version=MODEL_VERSION).set(1)
    MODEL_INFO.info({
        "version": MODEL_VERSION,
        "path": str(MODEL_PATH),
        "parameters": str(sum(p.numel() for p in model.parameters())),
        "labels": ",".join(LABEL_LIST),
    })

    logger.info(f"Both models loaded: student={MODEL_VERSION}, teacher={TEACHER_VERSION}")
    yield

    MODEL_LOADED.labels(model_version=MODEL_VERSION).set(0)
    logger.info("Server shutting down")


app = FastAPI(
    title="Clinical NER Inference API",
    description="PHI detection endpoint with full observability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/app", StaticFiles(directory="static", html=True), name="static")


@app.post("/predict", response_model=NERResponse)
async def predict(request: NERRequest):
    request_id = request.request_id or generate_request_id()

    ACTIVE_REQUESTS.inc()

    with tracer.start_as_current_span("ner_inference") as span:
        span.set_attribute("request_id", request_id)
        span.set_attribute("model_version", MODEL_VERSION)

        try:
            start_time = time.perf_counter()

            with tracer.start_as_current_span("tokenization"):
                inputs = tokenizer(
                    request.text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                    padding=True,
                )
                num_tokens = inputs["input_ids"].shape[1]
                INPUT_LENGTH.labels(model_version=MODEL_VERSION).observe(num_tokens)
                span.set_attribute("input_tokens", num_tokens)

            with tracer.start_as_current_span("model_inference"):
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

            with tracer.start_as_current_span("postprocessing"):
                predictions = torch.argmax(logits, dim=-1).squeeze().numpy()
                probabilities = torch.softmax(logits, dim=-1).squeeze().numpy()

                entities = []
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
                current_entity = None

                for idx, (token, pred_id) in enumerate(zip(tokens, predictions)):
                    if token in ["[CLS]", "[SEP]", "[PAD]"]:
                        continue

                    label = LABEL_LIST[pred_id]
                    confidence = float(probabilities[idx][pred_id])

                    CONFIDENCE_SCORE.labels(model_version=MODEL_VERSION).observe(confidence)

                    if label.startswith("B-"):
                        if current_entity:
                            entities.append(current_entity)
                        entity_type = label[2:]
                        current_entity = Entity(
                            text=token.replace("##", ""),
                            label=entity_type,
                            start=idx,
                            end=idx,
                            confidence=round(confidence, 4),
                        )
                        ENTITY_COUNT.labels(
                            entity_type=entity_type,
                            model_version=MODEL_VERSION,
                        ).inc()

                    elif label.startswith("I-") and current_entity:
                        current_entity.text += token.replace("##", "")
                        current_entity.end = idx
                        current_entity.confidence = round(
                            min(current_entity.confidence, confidence), 4
                        )

                    else:
                        if current_entity:
                            entities.append(current_entity)
                            current_entity = None

                if current_entity:
                    entities.append(current_entity)

            latency_ms = (time.perf_counter() - start_time) * 1000

            REQUEST_COUNT.labels(model_version=MODEL_VERSION, status="success").inc()
            REQUEST_LATENCY.labels(model_version=MODEL_VERSION).observe(latency_ms / 1000)

            if latency_ms > SLA_LATENCY_THRESHOLD_MS:
                SLA_VIOLATIONS.labels(model_version=MODEL_VERSION).inc()
            else:
                REQUESTS_WITHIN_SLA.labels(model_version=MODEL_VERSION).inc()

            entity_counts = {}
            for e in entities:
                entity_counts[e.label] = entity_counts.get(e.label, 0) + 1

            log_inference(
                logger, request_id, MODEL_VERSION, latency_ms,
                num_tokens, entity_counts,
            )

            ACTIVE_REQUESTS.dec()

            return NERResponse(
                request_id=request_id,
                entities=entities,
                latency_ms=round(latency_ms, 2),
                model_version=MODEL_VERSION,
                num_tokens=num_tokens,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            REQUEST_COUNT.labels(model_version=MODEL_VERSION, status="error").inc()
            ERROR_COUNT.labels(model_version=MODEL_VERSION, error_type=type(e).__name__).inc()
            log_error(logger, request_id, MODEL_VERSION, type(e).__name__, str(e))
            ACTIVE_REQUESTS.dec()
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


def run_ner(text, ner_model, ner_tokenizer, model_version):
    start_time = time.perf_counter()

    inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH, padding=True)
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())

    with torch.no_grad():
        outputs = ner_model(**inputs)
        logits = outputs.logits

    predictions = torch.argmax(logits, dim=-1).squeeze().numpy()
    probabilities = torch.softmax(logits, dim=-1).squeeze().numpy()

    entities = []
    current_entity = None

    for idx, (token, pred_id) in enumerate(zip(tokens, predictions)):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        label = LABEL_LIST[pred_id]
        confidence = float(probabilities[idx][pred_id])

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = Entity(
                text=token.replace("##", ""),
                label=label[2:],
                start=idx, end=idx,
                confidence=round(confidence, 4),
            )
        elif label.startswith("I-") and current_entity:
            current_entity.text += token.replace("##", "")
            current_entity.end = idx
            current_entity.confidence = round(min(current_entity.confidence, confidence), 4)
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    latency_ms = (time.perf_counter() - start_time) * 1000
    return entities, round(latency_ms, 2), len(tokens)


@app.post("/compare")
async def compare(request: NERRequest):
    request_id = request.request_id or generate_request_id()

    student_entities, student_latency, num_tokens = run_ner(text=request.text, ner_model=model, ner_tokenizer=tokenizer, model_version=MODEL_VERSION)
    teacher_entities, teacher_latency, _ = run_ner(text=request.text, ner_model=teacher_model, ner_tokenizer=teacher_tokenizer, model_version=TEACHER_VERSION)

    return {
        "request_id": request_id,
        "text": request.text,
        "student": {
            "model_version": MODEL_VERSION,
            "entities": [e.dict() for e in student_entities],
            "latency_ms": student_latency,
            "num_entities": len(student_entities),
            "parameters": "65.2M",
            "size": "249MB",
        },
        "teacher": {
            "model_version": TEACHER_VERSION,
            "entities": [e.dict() for e in teacher_entities],
            "latency_ms": teacher_latency,
            "num_entities": len(teacher_entities),
            "parameters": "107.7M",
            "size": "411MB",
        },
        "num_tokens": num_tokens,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "not_ready",
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
