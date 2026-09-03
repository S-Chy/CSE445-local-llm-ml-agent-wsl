"""
ml_tools.py
Baseline Machine Learning tools for the CSE445 Autonomous LLM ML Agent.
Each function returns a JSON string so it can be dropped straight into
an Observation in the ReAct loop.
"""
 
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
 
import torch
import torch.nn as nn
import torch.optim as optim
 
DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}
 
 
def load_dataset_summary(dataset_name: str) -> str:
    """Loads a standard benchmark dataset and returns summary statistics."""
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Unknown dataset '{name}'. Options: {list(DATASETS.keys())}"})
 
    data = DATASETS[name]()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
 
    summary = {
        "dataset": name,
        "n_samples": df.shape[0],
        "n_features": len(data.feature_names),
        "feature_names": list(data.feature_names),
        "classes": [str(c) for c in np.unique(data.target)],
        "missing_values": int(df.isnull().sum().sum()),
    }
    return json.dumps(summary)
 
 
def train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> str:
    """Trains a Scikit-Learn model (decision_tree, logistic_regression, random_forest) on a dataset."""
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found. Options: {list(DATASETS.keys())}"})
 
    if not (0.05 <= test_size <= 0.5):
        return json.dumps({"error": f"test_size must be between 0.05 and 0.5, got {test_size}"})
 
    data = DATASETS[name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=42, stratify=data.target
    )
 
    model_type = model_type.lower().strip()
    if model_type == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        return json.dumps({
            "error": f"Unsupported model '{model_type}'. "
                     f"Options: decision_tree, logistic_regression, random_forest"
        })
 
    try:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cv_scores = cross_val_score(clf, data.data, data.target, cv=5)
    except Exception as e:
        return json.dumps({"error": f"Training failed: {str(e)}"})
 
    return json.dumps({
        "model": model_type,
        "dataset": name,
        "test_accuracy": round(float(acc), 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
    })
 
 
def train_pytorch_mlp(dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01) -> str:
    """Trains a PyTorch Multilayer Perceptron on the selected classification dataset."""
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found. Options: {list(DATASETS.keys())}"})
 
    if hidden_dim < 1:
        return json.dumps({"error": f"hidden_dim must be a positive integer, got {hidden_dim}"})
    if epochs < 1:
        return json.dumps({"error": f"epochs must be a positive integer, got {epochs}"})
    if lr <= 0:
        return json.dumps({"error": f"lr must be positive, got {lr}"})
 
    data = DATASETS[name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
 
    # Feature standardization
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
 
    num_features = X_train.shape[1]
    num_classes = len(np.unique(data.target))
 
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.long)
 
    model = nn.Sequential(
        nn.Linear(num_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
 
    try:
        for _ in range(epochs):
            optimizer.zero_grad()
            out = model(X_t)
            loss = criterion(out, y_t)
            if torch.isnan(loss):
                return json.dumps({"error": "Training diverged: loss became NaN. Try a smaller lr."})
            loss.backward()
            optimizer.step()
    except Exception as e:
        return json.dumps({"error": f"Training failed: {str(e)}"})
 
    with torch.no_grad():
        test_out = model(X_val_t)
        test_preds = torch.argmax(test_out, dim=1)
        acc = (test_preds == y_val_t).float().mean().item()
 
    return json.dumps({
        "framework": "PyTorch",
        "dataset": name,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "final_loss": round(float(loss.item()), 4),
        "test_accuracy": round(float(acc), 4),
    })
 
 
# Registry mapping tool names to callable Python functions.
# react_agent.py imports this dict to know what tools are available.
AVAILABLE_TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
}