import argparse
import json
import os
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import (
    read_benchmark_report,
    read_all_reports,
    analyze_regressions,
    generate_evaluation_summary,
    check_alert_conditions,
    analyze_clinical_document,
    analyze_document_file,
)

RESULTS_DIR = Path("results")


@tool
def read_report(report_path: str = "") -> str:
    """Read a specific benchmark report JSON file. Leave path empty for the optimization report."""
    return read_benchmark_report(report_path)


@tool
def read_all() -> str:
    """Read all available benchmark reports from Components 1-3."""
    return read_all_reports()


@tool
def analyze() -> str:
    """Analyze all reports for performance regressions against defined thresholds."""
    return analyze_regressions()


@tool
def summarize() -> str:
    """Generate a Markdown evaluation summary from all reports."""
    return generate_evaluation_summary()


@tool
def check_alerts() -> str:
    """Check if any metrics breach alert thresholds."""
    return check_alert_conditions()


@tool
def analyze_document(text: str) -> str:
    """Run NER on clinical text to detect Chemical and Disease entities."""
    return analyze_clinical_document(text)


@tool
def analyze_file(file_path: str) -> str:
    """Read a text file and run NER analysis on its contents."""
    return analyze_document_file(file_path)


TOOLS = [read_report, read_all, analyze, summarize, check_alerts, analyze_document, analyze_file]


SYSTEM_PROMPT = """You are an ML Pipeline Evaluation Agent for a clinical NLP project.

Your job is to analyze benchmark results from an ML optimization pipeline and provide
a comprehensive evaluation. The pipeline includes:

1. Knowledge Distillation (Bio_ClinicalBERT → DistilBERT for clinical NER)
2. Distributed Data Processing (PySpark on AWS EMR)
3. Model Optimization (Pruning + INT8 Quantization for edge deployment)

When asked to evaluate the pipeline, follow these steps:
1. First, read all available reports to understand the current state
2. Run regression analysis to check for threshold breaches
3. Generate a human-readable summary
4. Check alert conditions

Be specific about numbers. If a metric is close to a threshold, call it out.
If the speedup is low due to hardware (macOS vs x86), explain that context.
Always frame findings in terms of clinical deployment readiness.
"""


def create_agent(provider="openrouter", model="nvidia/nemotron-3-super-120b-a12b:free"):
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Get a free key at https://openrouter.ai/keys\n"
                "  export OPENROUTER_API_KEY=your-key"
            )

        llm = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys\n"
                "  export GROQ_API_KEY=your-key"
            )

        llm = ChatGroq(model=model, temperature=0, api_key=api_key)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Either:\n"
                "  export OPENAI_API_KEY=your-key\n"
                "  or use --provider openrouter (free)"
            )

        llm = ChatOpenAI(model=model, temperature=0)

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        llm = ChatOllama(model=model, temperature=0)

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openrouter', 'groq', 'openai', or 'ollama'.")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )

    return executor


def run_evaluation(executor):
    print("ML Pipeline Evaluation Agent\n")

    result = executor.invoke({
        "input": (
            "Please perform a complete evaluation of the ML pipeline. "
            "Read all available reports, analyze for regressions, "
            "generate an evaluation summary, and check if any alerts "
            "need to be triggered. Provide your assessment of the "
            "pipeline's readiness for clinical deployment."
        )
    })

    print(f"\n{result['output']}")
    return result


def run_interactive(executor):
    print("Interactive mode — ask questions about the pipeline")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        result = executor.invoke({"input": user_input})
        print(f"\nAgent: {result['output']}")


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline Evaluation Agent")
    parser.add_argument("--provider", default="openrouter", choices=["openrouter", "groq", "openai", "ollama"],
                        help="LLM provider (default: openrouter)")
    parser.add_argument("--model", default="nvidia/nemotron-3-super-120b-a12b:free",
                        help="Model name (default: nvidia/nemotron-3-super-120b-a12b:free)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive Q&A mode")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    executor = create_agent(provider=args.provider, model=args.model)

    if args.interactive:
        run_interactive(executor)
    else:
        run_evaluation(executor)


if __name__ == "__main__":
    main()
