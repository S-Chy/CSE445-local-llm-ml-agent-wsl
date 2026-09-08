# CSE445 Local LLM ML Agent (WSL)
**Student:** Sajiah Islam Chowdhury  
**ID:** 2212447642
**Course:** CSE445 — Machine Learning, Section 6  

An autonomous, fully local Machine Learning agent — a quantized LLM served via
[Ollama](https://ollama.com) reasons through a ReAct (Reason + Act) loop, calling Python tools
that train and evaluate Scikit-Learn and PyTorch models. No external paid APIs are used.

## Architecture

| Component | Technology | Role |
|---|---|---|
| OS / Subsystem | Windows WSL2 (Ubuntu) | Native Linux execution environment |
| Inference Engine | Ollama (local REST API, port 11434) | Serves a quantized local model (`llama3.2:1b`) |
| Agent Logic | Python 3.11 (via pyenv) | Custom ReAct controller — prompts the model, parses actions, executes tools |
| ML Frameworks | PyTorch, Scikit-Learn, Pandas, NumPy | Data preprocessing, model training, evaluation |

## Project structure

```
CSE445-local-llm-ml-agent-wsl/
├── ml_tools.py            # 6 ML tools: dataset summary, sklearn training, PyTorch MLP,
│                          # hyperparameter tuning, PCA/feature selection, regularized MLP
├── react_agent.py         # ReAct loop controller — talks to Ollama, parses actions,
│                          # executes tools, with self-healing + repetition detection
├── benchmark_runner.py    # 3-algorithm x 2-dataset cross-validation benchmark
├── requirements.txt       # Python dependencies
├── execution_logs.txt     # 3 recorded multi-step agent reasoning traces
├── benchmark_results.md   # Markdown benchmark results table
├── technical_report.md    # Full technical report (architecture, prompt engineering,
│                          # latency benchmarks, model comparison, controller flow)
├── technical_note.md      # Engineering log of issues found and fixed during development
└── README.md
```

## Setup

```bash
# Python 3.11 environment (via pyenv)
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install and run Ollama, pull the model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
```

## Running

```bash
# Run the ReAct agent on the default test task
python react_agent.py

# Run the model comparison benchmark
python benchmark_runner.py
```

## Status

- [x] Task 1 — Environment setup (WSL2, Python 3.11, Ollama, PyTorch/Scikit-Learn), baseline ReAct loop
- [x] Task 2 — Advanced tools: hyperparameter tuning (GridSearchCV), PCA/feature selection, regularized PyTorch MLP (Dropout + BatchNorm + LR scheduler)
- [x] Task 3 — Self-healing (repetition detection + tool error surfacing), 3-algorithm x 2-dataset benchmark with Markdown summary
- [x] Execution logs (3 multi-step traces)
- [x] Technical report

See `technical_report.md` for full architecture details, latency benchmarks, and the
mathematical model comparison.