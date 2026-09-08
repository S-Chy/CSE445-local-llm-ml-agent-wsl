# Technical Report — CSE445 Assignment 3
**Building an Autonomous Local LLM Machine Learning Agent in Windows WSL**

---

## 1. Local LLM Architecture

The system runs entirely offline inside Windows WSL2 (Ubuntu), with no external paid APIs.

- **Inference engine:** [Ollama](https://ollama.com), serving a quantized model over its local
  REST API at `http://127.0.0.1:11434/api/generate`. Ollama manages model loading, quantization,
  and inference scheduling; the agent communicates with it purely over HTTP.
- **Agent controller (`react_agent.py`):** a Python ReAct (Reason + Act) loop. On each iteration
  it (1) sends the running conversation to Ollama, (2) parses the model's response for a
  `Thought / Action / Action Input` block using regular expressions, (3) executes the matching
  Python tool with the parsed arguments, (4) appends the real result as an `Observation`, and
  repeats until the model emits a `Final Answer`.
- **Tool layer (`ml_tools.py`):** six callable functions the agent can invoke — dataset
  summarization, three baseline model trainers (decision tree, logistic regression, random
  forest, and a PyTorch MLP), plus three advanced tools added for Task 2 (GridSearchCV-based
  hyperparameter tuning, PCA/sequential feature selection, and a regularized PyTorch MLP with
  Dropout, BatchNorm, and a StepLR scheduler). Every tool returns a JSON string, so results can
  be inserted directly into the conversation as an Observation.

## 2. Prompt Engineering Techniques

Two prompt-engineering problems had to be solved empirically, not just designed on paper.

**a) Format enforcement.** The system prompt defines a strict `Thought/Action/Action
Input`→`Final Answer` grammar and explicitly lists each tool's name, parameters, and return
type, e.g.:

```
train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2)
    -> JSON test/CV scores (model_type: decision_tree, logistic_regression, random_forest)
```

Listing exact parameter names and types (rather than a vague natural-language description)
measurably reduced the frequency of `TypeError`s from missing or misnamed arguments, since the
model is quoting the interface almost verbatim rather than paraphrasing it. Ollama's `stop`
parameter was also set to `["Observation:"]`, intending to halt generation right before the
model could fabricate a fake tool result.

**b) Truncation as a second line of defense.** In practice, the smaller local model did not
reliably respect the stop condition — it sometimes generated an entire fabricated multi-step
conversation (fake Actions, fake Observations with invented numbers, and a fake Final Answer)
in a single response, without the stop token ever appearing. This is a known limitation of
sub-2B-parameter instruction-tuned models: they can imitate the *shape* of a multi-turn
transcript convincingly without actually respecting the turn boundary they were asked to stop
at. To handle this, `react_agent.py` post-processes every raw response with
`truncate_to_first_step()`, which cuts the text at the first complete `Action`/`Action Input`
pair or the first `Final Answer` — whichever comes first — and discards anything generated
afterward. This ensures the agent only ever trusts and acts on one genuine step at a time,
regardless of what the model imagines happening next.

**c) Self-correction prompting.** Two further mechanisms close the loop between the model's
imperfect reasoning and reliable execution:
- *Repetition detection:* a set of `(tool, arguments)` signatures already executed is tracked;
  if the model requests an identical call again, it receives an Observation explicitly stating
  the action was already run, nudging it toward a different action or a Final Answer. This was
  necessary in practice — during development, the 1B model was observed calling
  `train_sklearn_model` on `random_forest` four consecutive times with identical arguments,
  apparently losing track of prior steps in a longer context window.
- *Error surfacing:* if a tool's JSON result contains an `"error"` key (bad parameters, an
  unsupported dataset/model name, or a NaN loss during training), the Observation explicitly
  says `ERROR — <message>. Reconsider your parameters...`, prompting the model to reason about
  the failure and retry with corrected arguments rather than treating the error as a normal
  result to report back to the user.

## 3. Latency Benchmarks in WSL2

Benchmarked directly on this machine (Windows host, 8GB total RAM, WSL2 Ubuntu):

| Model | Single short generation (`time ollama run ... "Say hello..."`) |
|---|---|
| `llama3.2:3b` | 74.6 s |
| `llama3.2:1b` | 20.5 s |

**Finding:** at 8GB total system RAM (WSL2 allocating roughly half by default), the 3B model's
per-call latency made an interactive multi-step agent loop impractical — a single task
requiring 4-6 reasoning steps would take several minutes purely in inference time, before
accounting for the model needing to reconsider on errors or repeated actions. The 1B model
reduced this to a usable range, at the cost of weaker instruction-following (motivating the
truncation and repetition-detection mechanisms in Section 2). This is a direct, measured
illustration of the classic latency-vs-capability trade-off in serving LLMs on resource
constrained hardware, rather than a purely theoretical concern.

## 4. Mathematical Comparison of Evaluated Models (CO1, CO2)

### 4.1 Mathematical Foundations of Each Algorithm

**Logistic Regression** models the probability of the positive class using the sigmoid function
applied to a linear combination of features:

$$P(y=1 \mid x) = \frac{1}{1 + e^{-(w^T x + b)}}$$

where $w$ is the learned weight vector and $b$ the bias term. Its decision boundary is linear in
feature space, and the model is fit by minimizing log-loss with an L2 regularization penalty by
default in Scikit-Learn, which controls overfitting by shrinking large weights.

**Decision Tree** classifiers greedily choose, at each node, the feature and threshold that most
reduce class impurity, most commonly measured by Gini impurity:

$$G = 1 - \sum_{j=1}^{C} p_j^2$$

where $p_j$ is the proportion of samples belonging to class $j$ at that node. A pure node (all
samples of one class) has $G = 0$; a maximally mixed binary node has $G = 0.5$. This tool
constrains `max_depth=4` to limit overfitting on small datasets.

**Random Forest** is an ensemble of decision trees, each trained on a bootstrap-resampled subset
of the training data and considering a random subset of features at each split. Its prediction
is the majority vote across trees. Because each tree's errors are only partially correlated with
the others' (due to the randomness in both sampling and feature selection), averaging over many
trees reduces prediction variance relative to a single tree, at some cost in interpretability.

`benchmark_runner.py` evaluates all three algorithms across two datasets (`wine`,
`breast_cancer`) using 5-fold cross-validation, computing the mean and standard deviation of
accuracy across folds as:

$$\bar{a} = \frac{1}{5}\sum_{i=1}^{5} a_i \qquad s = \sqrt{\frac{1}{5}\sum_{i=1}^{5}(a_i - \bar{a})^2}$$

### 4.2 Cross-Validation Results

| Dataset | Model | Test Accuracy | CV Mean Accuracy | CV Std |
|---|---|---|---|---|
| wine | decision_tree | 0.9444 | 0.8937 | 0.0472 |
| wine | logistic_regression | 0.9722 | 0.9611 | 0.0333 |
| wine | random_forest | 1.0 | 0.9610 | 0.0221 |
| breast_cancer | decision_tree | 0.9386 | 0.9209 | 0.0202 |
| breast_cancer | logistic_regression | 0.9561 | 0.9543 | 0.0102 |
| breast_cancer | random_forest | 0.9561 | 0.9543 | 0.0244 |

**Interpretation (CO2 — statistical/mathematical analysis):**
- On **wine**, `random_forest` achieves the highest single test-split accuracy (1.0) but its
  cross-validated mean (0.9610) is nearly identical to `logistic_regression`'s (0.9611); the
  single test-split number for random forest is likely an optimistic artifact of that
  particular 20% split rather than a genuinely superior model — this is precisely why CV mean
  and std, not a single test score, are the trustworthy comparison metric.
- `logistic_regression` has the **lowest variance** across folds on both datasets (std 0.0333
  and 0.0102 respectively), indicating more stable, reproducible performance than the
  tree-based models, whose variance (up to 0.0472 for `decision_tree` on wine) reflects greater
  sensitivity to which samples land in each fold.
- **decision_tree** consistently underperforms both other models on mean accuracy across both
  datasets, consistent with it being the least ensembled/regularized of the three — it has no
  variance-reduction mechanism (unlike random forest's bagging) and no margin-based
  regularization (unlike logistic regression's L2 penalty).

**Recommendation:** for a deployment scenario prioritizing predictable, low-variance behavior
(e.g., clinical screening), `logistic_regression`'s combination of high mean accuracy and low
standard deviation makes it the more defensible choice over `random_forest` despite the latter's
higher peak test-set number, illustrating CO1's emphasis on understanding *why* an algorithm
performs the way it does, not just which number is largest.

## 5. Architecture Diagram — Agent Controller Loop and Tool Registry

The controller loop in `react_agent.py` follows this flow on every iteration:

1. **User Query** enters `run_agent_loop()`.
2. **`query_local_llm()`** sends the running prompt/history to the Ollama REST API
   (`127.0.0.1:11434`, model `llama3.2:1b`) and receives a raw response.
3. **`truncate_to_first_step()`** cuts that response at the first real `Action`/`Action Input`
   pair or the first `Final Answer` — discarding any hallucinated continuation the model
   generated past that point.
4. **`parse_action()`** extracts the tool name and arguments (if any).
   - If the response is a **Final Answer** → the loop ends and the answer is returned to the user.
   - If an **Action** was requested → continue to step 5.
5. **Repetition check**: the `(tool, arguments)` signature is compared against
   `executed_actions`. If it's a duplicate, the model receives an Observation telling it to try
   something different instead of re-running the tool.
6. **`execute_tool()`** calls the matching function in `ml_tools.py`'s `AVAILABLE_TOOLS`
   registry: `load_dataset_summary`, `train_sklearn_model`, `train_pytorch_mlp`,
   `tune_hyperparameters`, `reduce_dimensionality`, or `train_regularized_mlp`.
7. **Self-healing check**: if the tool's JSON result contains an `"error"` key, the Observation
   explicitly flags it (`ERROR — ...`) and asks the model to reconsider its parameters.
8. The **Observation** (real result or error) is appended to the prompt, and the loop returns to
   step 2 — repeating until a genuine Final Answer is produced or `max_iterations` is reached.

## 6. Summary of Deliverables

| Task | Status |
|---|---|
| Task 1 — Environment, baseline ReAct loop | Complete (CPU-only PyTorch; GPU passthrough out of scope) |
| Task 2 — Hyperparameter tuning, PCA/feature selection, regularized MLP | Complete, all three tools verified |
| Task 3 — Self-healing (repetition + error detection), benchmark | Complete; benchmark implemented as a deterministic script (`benchmark_runner.py`) calling the same `ml_tools` functions the agent uses, given the latency figures in Section 3 |

See `technical_note.md` for the original, narrower engineering log this report was expanded
from, and `benchmark_results.md` / `execution_logs.txt` for raw supporting output.

## 7. Limitations

The system relies on a small local model (`llama3.2:1b`), which trades reasoning quality for
latency — it more frequently makes formatting mistakes, loses track of prior steps, or omits
required arguments compared to larger models, which is why the truncation, repetition-detection,
and error-surfacing mechanisms in Section 2 were necessary rather than optional. The agent's
available actions are limited to the six tools explicitly registered in `ml_tools.py`; it cannot
invent new operations. All PyTorch training in this project runs on CPU inside WSL2 — no GPU
passthrough was configured, which is a hardware/scope constraint rather than a design choice,
and would meaningfully reduce training time for larger models or datasets if available. Finally,
because inference latency accumulates across steps, the deterministic `benchmark_runner.py`
script was used for the full 3-algorithm × 2-dataset comparison rather than driving all six
combinations through the LLM step-by-step, trading some autonomy for reproducibility within the
available time.

## 8. Conclusion

This project demonstrates that a small, quantized language model running entirely offline in
Windows WSL2 via Ollama can function as a working Machine Learning agent when paired with a
carefully engineered ReAct controller. The main engineering challenges were not in defining the
ML tools themselves — Scikit-Learn and PyTorch handle that reliably — but in compensating for
the local model's unreliable adherence to the ReAct protocol: hallucinated multi-step traces,
repeated identical actions, and malformed tool arguments. Addressing each of these with a
targeted, testable mechanism (truncation, repetition detection, and error surfacing,
respectively) turned an initially unreliable loop into one that consistently executes real
tools and returns genuine results, while keeping the entire pipeline free of external paid APIs.
