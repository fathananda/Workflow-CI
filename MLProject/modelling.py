"""
Modelling Script untuk MLProject - CI Workflow
Nama   : Fathi Ananda Mas'ud
Email  : fathiananda00@gmail.com
Dataset: Iris Dataset (Preprocessed)
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# ─── Argument Parser ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Iris Classification MLProject")
parser.add_argument("--dataset_path", type=str, default="iris_preprocessing.csv")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--min_samples_split", type=int, default=2)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# ─── Load Data ─────────────────────────────────────────────────────────────────
print(f"[INFO] Loading dataset: {args.dataset_path}")
df = pd.read_csv(args.dataset_path)
print(f"[INFO] Shape: {df.shape}")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=args.test_size,
    random_state=args.random_state,
    stratify=y
)
print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

# ─── MLflow Tracking ───────────────────────────────────────────────────────────
# ─── MLflow Tracking ───────────────────────────────────────────
if mlflow.active_run() is None:
    mlflow.set_experiment("Iris_CI_Workflow_Fathi")
    active_run = mlflow.start_run()
else:
    active_run = mlflow.active_run()

print(f"[INFO] Run ID: {active_run.info.run_id}")

# Log Parameters
mlflow.log_param("n_estimators", args.n_estimators)
mlflow.log_param("max_depth", args.max_depth)
mlflow.log_param("min_samples_split", args.min_samples_split)
mlflow.log_param("test_size", args.test_size)
mlflow.log_param("random_state", args.random_state)

# Training
model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    min_samples_split=args.min_samples_split,
    random_state=args.random_state
)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall    = recall_score(y_test, y_pred, average="weighted")
f1        = f1_score(y_test, y_pred, average="weighted")

print(f"\n[HASIL] Accuracy : {accuracy:.4f}")
print(f"[HASIL] Precision: {precision:.4f}")
print(f"[HASIL] Recall   : {recall:.4f}")
print(f"[HASIL] F1-Score : {f1:.4f}")

# Log Metrics
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision_weighted", precision)
mlflow.log_metric("recall_weighted", recall)
mlflow.log_metric("f1_score_weighted", f1)

# Log Model
mlflow.sklearn.log_model(model, artifact_path="model")

# Confusion Matrix Artifact
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Setosa", "Versicolor", "Virginica"],
            yticklabels=["Setosa", "Versicolor", "Virginica"], ax=ax)
ax.set_title("Confusion Matrix - Fathi Ananda Mas'ud")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
cm_path = "confusion_matrix.png"
plt.savefig(cm_path)
plt.close()
mlflow.log_artifact(cm_path)
if os.path.exists(cm_path):
    os.remove(cm_path)

print("\n[INFO] Training selesai!")
