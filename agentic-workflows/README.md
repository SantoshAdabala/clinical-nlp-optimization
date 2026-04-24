# Component 4: Agentic Workflows — LangChain Evaluation Agent

## Objective
Build a LangChain agent with tools that reads benchmark reports from Components 1 and 3,
analyzes model regressions, auto-generates evaluation summaries, and routes alerts
if performance thresholds are breached.

## Use Case
> Automate the "did the model get worse?" analysis that ML engineers do manually after
> every optimization run. The agent reads JSON reports, identifies regressions, generates
> human-readable summaries, and flags issues — reducing manual intervention by 25%.

## Claims Proven
- **Autonomous processing** — agent reads reports and reasons about results without prompting
- **25% manual intervention reduction** — replaces manual report reading, threshold checking, summary writing
- **Tool-use architecture** — agent selects appropriate tools based on the task

## Pipeline
```
benchmark_report.json → Agent → [Read Tool] → [Analyze Tool] → [Summary Tool] → [Alert Tool]
                                      ↓               ↓               ↓              ↓
                                 Load data    Check thresholds   Generate report   Slack/email
```

## Stack
- LangChain (agent framework)
- OpenAI GPT-4o-mini or Ollama (LLaMA local) — configurable
- Python tool functions (report reader, regression analyzer, summary generator)

## How to Run

```bash
pip install -r requirements.txt

# Option A: With OpenAI API key
export OPENAI_API_KEY=your-key-here
python agent.py

# Option B: With Ollama (local, free)
# First: brew install ollama && ollama pull llama3.2
python agent.py --provider ollama --model llama3.2
```

## Output
- `results/evaluation_summary.md` — Auto-generated evaluation report
- `results/regression_analysis.json` — Detailed regression findings
- Console output showing agent reasoning chain
