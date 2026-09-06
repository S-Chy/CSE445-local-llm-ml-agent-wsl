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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
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
 
 
# ---------------------------------------------------------------------------
# Task 2: Advanced ML Tools
# ---------------------------------------------------------------------------
 
def tune_hyperparameters(dataset_name: str, model_type: str) -> str:
    """Runs GridSearchCV for 'svc' or 'decision_tree' on the given dataset, returns best params/score."""
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found. Options: {list(DATASETS.keys())}"})
 
    data = DATASETS[name]()
    model_type = model_type.lower().strip()
 
    if model_type == "svc":
        estimator = SVC()
        param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
    elif model_type == "decision_tree":
        estimator = DecisionTreeClassifier(random_state=42)
        param_grid = {"max_depth": [2, 4, 6, None], "min_samples_split": [2, 5, 10]}
    else:
        return json.dumps({"error": f"Unsupported model '{model_type}'. Options: svc, decision_tree"})
 
    try:
        search = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1)
        search.fit(data.data, data.target)
    except Exception as e:
        return json.dumps({"error": f"Hyperparameter tuning failed: {str(e)}"})
 
    return json.dumps({
        "model": model_type,
        "dataset": name,
        "best_params": search.best_params_,
        "best_cv_accuracy": round(float(search.best_score_), 4),
    })
 
 
def reduce_dimensionality(dataset_name: str, method: str = "pca", n_components: int = 2) -> str:
    """Applies 'pca' or 'sequential' feature selection to a dataset and reports variance/accuracy impact."""
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found. Options: {list(DATASETS.keys())}"})
 
    data = DATASETS[name]()
    method = method.lower().strip()
 
    if n_components < 1:
        return json.dumps({"error": f"n_components must be a positive integer, got {n_components}"})
 
    try:
        if method == "pca":
            pca = PCA(n_components=n_components, random_state=42)
            reduced = pca.fit_transform(data.data)
            result = {
                "method": "pca",
                "dataset": name,
                "n_components": n_components,
                "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
                "total_variance_explained": round(float(sum(pca.explained_variance_ratio_)), 4),
            }
        elif method == "sequential":
            base_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
            selector = SequentialFeatureSelector(base_clf, n_features_to_select=n_components, cv=5)
            selector.fit(data.data, data.target)
            selected = [f for f, keep in zip(data.feature_names, selector.get_support()) if keep]
            result = {
                "method": "sequential",
                "dataset": name,
                "n_features_selected": n_components,
                "selected_features": selected,
            }
        else:
            return json.dumps({"error": f"Unsupported method '{method}'. Options: pca, sequential"})
    except Exception as e:
        return json.dumps({"error": f"Dimensionality reduction failed: {str(e)}"})
 
    return json.dumps(result)
 
 
class RegularizedMLP(nn.Module):
    """A configurable feed-forward classifier with Dropout and BatchNorm."""
 
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
 
    def forward(self, x):
        return self.net(x)
 
 
def train_regularized_mlp(dataset_name: str, hidden_dim: int = 32, epochs: int = 50,
                           lr: float = 0.01, dropout: float = 0.3) -> str:
    """Trains a PyTorch MLP with Dropout, BatchNorm, and a StepLR scheduler."""
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found. Options: {list(DATASETS.keys())}"})
    if hidden_dim < 2:
        return json.dumps({"error": f"hidden_dim must be at least 2, got {hidden_dim}"})
    if not (0 <= dropout < 1):
        return json.dumps({"error": f"dropout must be between 0 and 1, got {dropout}"})
 
    data = DATASETS[name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
 
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
 
    num_features = X_train.shape[1]
    num_classes = len(np.unique(data.target))
 
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.long)
 
    model = RegularizedMLP(num_features, hidden_dim, num_classes, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.5)
 
    try:
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = model(X_t)
            loss = criterion(out, y_t)
            if torch.isnan(loss):
                return json.dumps({"error": "Training diverged: loss became NaN. Try a smaller lr."})
            loss.backward()
            optimizer.step()
            scheduler.step()
    except Exception as e:
        return json.dumps({"error": f"Training failed: {str(e)}"})
 
    model.eval()
    with torch.no_grad():
        test_out = model(X_val_t)
        test_preds = torch.argmax(test_out, dim=1)
        acc = (test_preds == y_val_t).float().mean().item()
 
    return json.dumps({
        "framework": "PyTorch",
        "architecture": "Dropout + BatchNorm + StepLR",
        "dataset": name,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "epochs": epochs,
        "final_loss": round(float(loss.item()), 4),
        "final_lr": round(float(scheduler.get_last_lr()[0]), 6),
        "test_accuracy": round(float(acc), 4),
    })
 
 
# Registry mapping tool names to callable Python functions.
# react_agent.py imports this dict to know what tools are available.
AVAILABLE_TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
    "tune_hyperparameters": tune_hyperparameters,
    "reduce_dimensionality": reduce_dimensionality,
    "train_regularized_mlp": train_regularized_mlp,
}