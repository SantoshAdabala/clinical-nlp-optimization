# Component 6: Observability — Metrics, Logging, Tracing

## Objective
Add production observability to the clinical NER inference pipeline:
Prometheus metrics, structured JSON logging, OpenTelemetry distributed tracing,
and a Grafana dashboard for SLA monitoring.

## The Three Pillars
```
METRICS (Prometheus)          LOGS (Structured JSON)       TRACES (OpenTelemetry)
"How is it performing?"       "What happened?"             "Where did time go?"
- Request count               - Per-request details        - Tokenization: 2ms
- Latency histogram            - Errors with context        - Inference: 15ms
- Entity counts                - Model version              - Postprocess: 1ms
- Error rate                   - Input characteristics      - Total: 18ms
```

## Stack
- Prometheus client (Python metrics)
- OpenTelemetry (distributed tracing)
- Structured logging (JSON format)
- Grafana dashboard config (JSON provisioning)
- FastAPI (inference server with observability baked in)

## How to Run
```bash
pip install -r requirements.txt

# Run the instrumented inference server
python inference_server.py

# In another terminal — send test requests
python test_client.py

# View metrics at http://localhost:8000/metrics
# View health at http://localhost:8000/health
```

## Output
- `/metrics` endpoint — Prometheus-compatible metrics
- `/health` endpoint — Health check with model info
- `logs/inference.log` — Structured JSON logs
- `grafana/dashboard.json` — Import into Grafana
