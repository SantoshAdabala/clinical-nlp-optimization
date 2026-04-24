import json
from pathlib import Path
from tools import (
    read_all_reports,
    analyze_regressions,
    generate_evaluation_summary,
    check_alert_conditions,
)

RESULTS_DIR = Path("results")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Direct tool execution (no LLM required)\n")

    print("[1/4] Reading all reports...")
    reports = json.loads(read_all_reports())
    available = list(reports.keys())
    if "error" in reports:
        print(f"  ERROR: {reports['error']}")
        return
    print(f"  Found reports: {available}")

    print("\n[2/4] Analyzing regressions...")
    regressions = json.loads(analyze_regressions())
    print(f"  Overall status: {regressions['overall_status']}")
    for finding in regressions["findings"]:
        icon = "✅" if finding["severity"] == "OK" else "⚠️" if finding["severity"] == "MEDIUM" else "❌"
        print(f"  {icon} {finding['metric']}: {finding['message']}")

    print("\n[3/4] Generating evaluation summary...")
    summary = generate_evaluation_summary()
    print(f"  Saved to: {RESULTS_DIR / 'evaluation_summary.md'}")

    print("\n[4/4] Checking alert conditions...")
    alerts = json.loads(check_alert_conditions())
    print(f"  Alerts triggered: {alerts['alerts_triggered']}")
    print(f"  Action required: {alerts['action_required']}")
    for alert in alerts["alerts"]:
        print(f"    [{alert['severity']}] {alert['component']}: {alert['message']}")

    print(f"\n  Status: {regressions['overall_status']}")
    print(f"  Reports analyzed: {len(available)}")
    print(f"  Findings: {len(regressions['findings'])}")
    print(f"  Alerts: {alerts['alerts_triggered']}")
    print(f"\n  Full summary: {RESULTS_DIR / 'evaluation_summary.md'}")
    print(f"  Regression details: {RESULTS_DIR / 'regression_analysis.json'}")


if __name__ == "__main__":
    main()
