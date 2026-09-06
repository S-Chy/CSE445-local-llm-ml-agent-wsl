"""
react_agent.py
ReAct (Reason + Act) loop controller for the CSE445 Autonomous LLM ML Agent.
 
Sends a system prompt + running conversation to a local Ollama model,
parses the model's Thought/Action/Action Input output, executes the
matching tool from ml_tools.AVAILABLE_TOOLS, feeds the result back in
as an Observation, and repeats until the model produces a Final Answer.
"""
 
import re
import json
import requests
 
from ml_tools import AVAILABLE_TOOLS
 
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3.2:1b"
 
SYSTEM_PROMPT = """You are an expert Autonomous Machine Learning Assistant.
You solve machine learning problems by thinking step-by-step and invoking external tools.
 
You have access to the following tools:
1. load_dataset_summary(dataset_name: str) -> JSON summary of dataset (iris, wine, breast_cancer).
2. train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> JSON test/CV scores
   (model_type: decision_tree, logistic_regression, random_forest).
3. train_pytorch_mlp(dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01) -> JSON
   PyTorch training and evaluation results.
 
To use a tool, you MUST strictly use this exact format, with the Action Input as a single-line JSON object:
 
Thought: Describe your reasoning about what to do next.
Action: tool_name
Action Input: {"param_name": "value"}
 
When you have received the observation and are ready to give the complete answer to the user, respond with:
 
Thought: I have gathered all necessary experimental data.
Final Answer: <your full answer here>
 
Begin!
"""
 
# Matches "Action: tool_name" (letters, digits, underscores only)
ACTION_RE = re.compile(r"Action:\s*([a-zA-Z0-9_]+)")
# Matches "Action Input: {...}" — non-greedy, stops at the FIRST closing brace
# (our tool arguments are flat, so no nested braces to worry about)
ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL)
FINAL_ANSWER_RE = re.compile(r"Final Answer:")
 
 
def truncate_to_first_step(llm_output: str) -> str:
    """
    Small local models often ignore the 'stop and wait for a real Observation'
    instruction and instead hallucinate an entire fake multi-step trace —
    fake actions, fake observations, even a fake Final Answer — all in one
    response. This cuts the output at whichever comes FIRST: a complete
    Action/Action Input pair, or a Final Answer — and discards everything
    the model imagined happening afterward, so we only ever trust the model
    one real step at a time.
    """
    action_input_match = ACTION_INPUT_RE.search(llm_output)
    final_match = FINAL_ANSWER_RE.search(llm_output)
 
    action_end = action_input_match.end() if action_input_match else None
    final_start = final_match.start() if final_match else None
 
    if action_end is not None and (final_start is None or action_end <= final_start):
        # A real action request comes first (or is the only thing present) — keep just that.
        return llm_output[:action_end]
 
    if final_start is not None:
        # Final Answer comes first — keep it plus its answer text, but stop
        # before any further hallucinated "Thought:"/"Action:" the model
        # might have kept generating afterward.
        rest = llm_output[final_match.end():]
        next_thought = re.search(r"\bThought:", rest)
        end = final_match.end() + (next_thought.start() if next_thought else len(rest))
        return llm_output[:end]
 
    return llm_output  # no action and no final answer — return as-is, caller will nudge the model
 
 
def query_local_llm(prompt: str) -> str:
    """Sends a prompt to the local Ollama instance running in WSL and returns the raw text response."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "stop": ["Observation:"],
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama error ({response.status_code}): {response.text}")
    return response.json().get("response", "")
 
 
def parse_action(llm_output: str):
    """
    Extracts (tool_name, kwargs_dict) from the model's output.
    Returns (None, None, error_message) if parsing fails, so the caller can
    feed the error back to the model as an Observation instead of crashing.
    """
    action_match = ACTION_RE.search(llm_output)
    input_match = ACTION_INPUT_RE.search(llm_output)
 
    if not action_match:
        return None, None, None  # no action requested at all — not necessarily an error
 
    if not input_match:
        return action_match.group(1).strip(), None, "Action Input not found or malformed."
 
    tool_name = action_match.group(1).strip()
    raw_input = input_match.group(1).strip()
 
    try:
        kwargs = json.loads(raw_input)
    except json.JSONDecodeError as e:
        return tool_name, None, f"Action Input is not valid JSON: {e}"
 
    return tool_name, kwargs, None
 
 
def execute_tool(tool_name: str, kwargs: dict) -> str:
    """Runs the requested tool and always returns a string safe to insert as an Observation."""
    if tool_name not in AVAILABLE_TOOLS:
        available = ", ".join(AVAILABLE_TOOLS.keys())
        return f"Tool '{tool_name}' not recognized. Available tools: {available}"
 
    try:
        return AVAILABLE_TOOLS[tool_name](**kwargs)
    except TypeError as e:
        # Usually a wrong/missing argument name — the kind of error Task 3's
        # self-correction logic should catch and retry with fixed params.
        return f"Tool execution error (bad arguments): {e}"
    except Exception as e:
        return f"Tool execution error: {e}"
 
 
def run_agent_loop(user_query: str, max_iterations: int = 6, verbose: bool = True) -> str:
    """
    Runs the ReAct loop until the model produces a Final Answer or max_iterations
    is reached. Returns the full conversation transcript as a string.
    """
    if verbose:
        print("=" * 60)
        print(f"USER QUERY: {user_query}")
        print("=" * 60)
 
    prompt = f"{SYSTEM_PROMPT}\nUser Query: {user_query}\n"
    executed_actions = set()  # tracks (tool_name, sorted kwargs) already run, to catch repetition loops
 
    for step in range(1, max_iterations + 1):
        if verbose:
            print(f"\n--- Step {step} ---")
 
        llm_output = query_local_llm(prompt)
        llm_output = truncate_to_first_step(llm_output)
        if verbose:
            print(llm_output)
 
        prompt += llm_output
 
        if "Final Answer:" in llm_output:
            if verbose:
                print("\n>>> Task completed successfully.")
            break
 
        tool_name, kwargs, parse_error = parse_action(llm_output)
 
        if tool_name is None:
            # Model didn't request an action or give a final answer — nudge it.
            observation = "\nObservation: Please respond with an Action and Action Input, or a Final Answer.\n"
        elif parse_error:
            observation = f"\nObservation: {parse_error}\n"
        else:
            action_signature = (tool_name, tuple(sorted((kwargs or {}).items())))
            if action_signature in executed_actions:
                observation = (
                    "\nObservation: You already ran this exact action and got the same result. "
                    "Do not repeat it — move on to a different tool, or provide your Final Answer "
                    "if you have everything you need.\n"
                )
            else:
                executed_actions.add(action_signature)
                tool_result = execute_tool(tool_name, kwargs)
                # Self-healing: if the tool reported an error (bad params, shape
                # mismatch, NaN loss), make that explicit so the model reasons
                # about the failure and retries with corrected parameters,
                # instead of just seeing another opaque JSON blob.
                try:
                    parsed_result = json.loads(tool_result)
                    if isinstance(parsed_result, dict) and "error" in parsed_result:
                        observation = (
                            f"\nObservation: ERROR — {parsed_result['error']} "
                            f"Reconsider your parameters for '{tool_name}' and try again with corrected values.\n"
                        )
                    else:
                        observation = f"\nObservation: {tool_result}\n"
                except (json.JSONDecodeError, TypeError):
                    observation = f"\nObservation: {tool_result}\n"
 
        if verbose:
            print(observation)
        prompt += observation
    else:
        if verbose:
            print("\n>>> Max iterations reached without a Final Answer.")
 
    return prompt
 
 
if __name__ == "__main__":
    test_task = (
        "Analyze the breast_cancer dataset, train a Random Forest and a PyTorch MLP on it, "
        "compare their accuracies, and recommend the best model for clinical screening."
    )
    run_agent_loop(test_task, max_iterations=10)