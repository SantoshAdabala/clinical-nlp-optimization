from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    HAS_OTLP = True
except ImportError:
    HAS_OTLP = False


def setup_tracing(service_name="clinical-ner-inference", export_to_console=True,
                  otlp_endpoint=None):
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": "development",
    })

    provider = TracerProvider(resource=resource)

    if export_to_console:
        provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter())
        )

    if otlp_endpoint and HAS_OTLP:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))

    trace.set_tracer_provider(provider)

    return trace.get_tracer(service_name)


def get_tracer(name="clinical-ner-inference"):
    return trace.get_tracer(name)
