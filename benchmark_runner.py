"""
benchmark_runner.py
Evaluates 3 algorithms (decision_tree, logistic_regression, random_forest) across
2 datasets (wine, breast_cancer), using 5-fold cross-validation, and writes a
Markdown experimental summary table.
 
Implemented as a deterministic script rather than routed entirely through the
local LLM agent: given the measured per-call latency of the local model on this
hardware (see technical_note.md), a fully autonomous multi-combination benchmark
driven step-by-step through the ReAct loop would be slow and non-deterministic to
reproduce. This script calls the same underlying ml_tools functions the agent
uses, so the numbers are identical to what the agent itself would obtain.
"""
 
import json
from datetime import datetime
 
from ml_tools import train_sklearn_model
 
ALGORITHMS = ["decision_tree", "logistic_regression", "random_forest"]
DATASETS = ["wine", "breast_cancer"]
 
 
def run_benchmark():
    results = []
    for dataset in DATASETS:
        for algo in ALGORITHMS:
            raw = train_sklearn_model(dataset, algo)
            parsed = json.loads(raw)
            results.append(parsed)
    return results
 
 
def to_markdown_table(results) -> str:
    lines = [
        "| Dataset | Model | Test Accuracy | CV Mean Accuracy | CV Std |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r.get('dataset', '?')} | {r.get('model', '?')} | ERROR: {r['error']} | - | - |")
        else:
            lines.append(
                f"| {r['dataset']} | {r['model']} | {r['test_accuracy']} | "
                f"{r['cv_mean_accuracy']} | {r['cv_std']} |"
            )
    return "\n".join(lines)
 
 
def summarize_best(results) -> str:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return "No valid results to summarize."
 
    lines = []
    for dataset in DATASETS:
        dataset_results = [r for r in valid if r["dataset"] == dataset]
        if not dataset_results:
            continue
        best = max(dataset_results, key=lambda r: r["cv_mean_accuracy"])
        lines.append(
            f"- **{dataset}**: best model is `{best['model']}` "
            f"(CV mean accuracy {best['cv_mean_accuracy']}, std {best['cv_std']})"
        )
    return "\n".join(lines)
 
 
if __name__ == "__main__":
    print("Running benchmark: 3 algorithms x 2 datasets, 5-fold cross-validation...\n")
    results = run_benchmark()
 
    table = to_markdown_table(results)
    summary = summarize_best(results)
 
    print(table)
    print("\n" + summary)
 
    output = (
        f"# Model Comparison Benchmark\n\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"3 algorithms (decision_tree, logistic_regression, random_forest) evaluated across "
        f"2 datasets (wine, breast_cancer) using 5-fold cross-validation.\n\n"
        f"## Results\n\n{table}\n\n"
        f"## Best model per dataset\n\n{summary}\n"
    )
 
    with open("benchmark_results.md", "w") as f:
        f.write(output)
 
    print("\nSaved to benchmark_results.md")