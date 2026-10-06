"""Sign Language Model Evaluation & Testing Module.

Evaluates the trained RandomForest sign language recognition model on holdout test data
using standardized metrics: Accuracy, Precision, Recall, F1-Score (macro & weighted),
and generates a confusion matrix and classification reports.

Usage:
    python evaluate_model.py
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import joblib

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATASET_PATH = BASE_DIR / "dataset" / "sign_landmarks.csv"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "sign_language_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_JSON = REPORTS_DIR / "classification_report.json"
CLASSIFICATION_REPORT_CSV = REPORTS_DIR / "classification_report.csv"
EVALUATION_SUMMARY_JSON = REPORTS_DIR / "model_evaluation.json"

# Evaluation parameters (matching model training parameters for reproducible holdout split)
RANDOM_STATE = 42
TEST_SIZE = 0.20
LABEL_COLUMN = "label"


def print_banner(title: str) -> None:
    """Print a clean visual section header."""
    print("=" * 72)
    print(f" {title}")
    print("=" * 72)


def load_model_artifacts() -> Tuple[Any, Any, Any]:
    """Load model, scaler, and label encoder from disk."""
    missing = [p for p in (MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required model artifacts: {[str(p) for p in missing]}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, scaler, label_encoder


def load_and_preprocess_dataset() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Load CSV dataset and separate numerical features from labels."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found in dataset")

    feature_cols = [col for col in df.columns if col not in (LABEL_COLUMN, "hand_count")]
    X = df[feature_cols].values
    y = df[LABEL_COLUMN].values

    return df, X, y, feature_cols


def evaluate_model():
    """Run comprehensive model evaluation and generate reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print_banner("SIGN LANGUAGE MULTIMODAL AI — MODEL EVALUATION")
    print()

    # 1. Load Artifacts
    print("1. Loading Model Artifacts...")
    model, scaler, le = load_model_artifacts()
    classes = list(le.classes_)
    expected_features = getattr(scaler, "n_features_in_", 63)
    model_type = type(model).__name__

    print(f"   ✓ Model Loaded:       {model_type}")
    print(f"   ✓ Classes ({len(classes)}):       {', '.join(classes)}")
    print(f"   ✓ Input Features:     {expected_features}")
    print()

    # 2. Load Dataset
    print("2. Loading Dataset...")
    df, X, y, feature_cols = load_and_preprocess_dataset()
    total_samples = len(df)
    print(f"   ✓ Total Samples:      {total_samples}")
    print(f"   ✓ Feature Columns:    {len(feature_cols)}")
    print()

    # 3. Create Reproducible Holdout Test Split
    print("3. Generating Reproducible Holdout Split...")
    y_encoded = le.transform(y)

    X_train, X_test, y_train_enc, y_test_enc, y_train_raw, y_test_raw = train_test_split(
        X,
        y_encoded,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    test_samples = len(X_test)
    train_samples = len(X_train)
    print(f"   ✓ Train Samples:      {train_samples} ({(1 - TEST_SIZE) * 100:.0f}%)")
    print(f"   ✓ Evaluation Samples: {test_samples} ({TEST_SIZE * 100:.0f}%) [Held-out for evaluation]")
    print(f"   ✓ Random Seed:        {RANDOM_STATE} (Reproducible Stratified Split)")
    print()

    # 4. Feature Scaling (Using fitted scaler on holdout test features)
    print("4. Scaling Test Features...")
    X_test_scaled = scaler.transform(X_test)
    print("   ✓ Features transformed via StandardScaler")
    print()

    # 5. Model Inference on Test Set
    print("5. Running Inference...")
    y_pred_enc = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    y_pred_labels = le.inverse_transform(y_pred_enc)
    print("   ✓ Predictions generated")
    print()

    # 6. Calculate Standard Metrics
    accuracy = float(accuracy_score(y_test_enc, y_pred_enc))
    macro_precision = float(precision_score(y_test_enc, y_pred_enc, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test_enc, y_pred_enc, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test_enc, y_pred_enc, average="macro", zero_division=0))

    weighted_precision = float(precision_score(y_test_enc, y_pred_enc, average="weighted", zero_division=0))
    weighted_recall = float(recall_score(y_test_enc, y_pred_enc, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_test_enc, y_pred_enc, average="weighted", zero_division=0))

    # Per-Class Classification Report
    report_dict = classification_report(
        y_test_enc,
        y_pred_enc,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    # 7. Print Terminal Evaluation Summary
    print_banner("EVALUATION RESULTS (HOLDOUT TEST SET)")
    print()
    print(f"  Overall Accuracy:    {accuracy * 100:.2f}%")
    print()
    print(f"  Macro Precision:     {macro_precision * 100:.2f}%")
    print(f"  Macro Recall:        {macro_recall * 100:.2f}%")
    print(f"  Macro F1-Score:      {macro_f1 * 100:.2f}%")
    print()
    print(f"  Weighted Precision:  {weighted_precision * 100:.2f}%")
    print(f"  Weighted Recall:     {weighted_recall * 100:.2f}%")
    print(f"  Weighted F1-Score:   {weighted_f1 * 100:.2f}%")
    print()

    # Per-Class Table
    print_banner("PER-SIGN PERFORMANCE BREAKDOWN")
    print(f"  {'SIGN':<16} {'PRECISION':<12} {'RECALL':<12} {'F1-SCORE':<12} {'SAMPLES':<8}")
    print("  " + "-" * 62)

    per_class_f1 = {}
    for cls_name in classes:
        metrics = report_dict.get(cls_name, {})
        prec = metrics.get("precision", 0.0) * 100
        rec = metrics.get("recall", 0.0) * 100
        f1 = metrics.get("f1-score", 0.0) * 100
        supp = int(metrics.get("support", 0))
        per_class_f1[cls_name] = f1
        print(f"  {cls_name:<16} {prec:>8.2f}%   {rec:>8.2f}%   {f1:>8.2f}%   {supp:>6}")

    print("  " + "-" * 62)
    print()

    # Determine Best and Weakest Performing Signs
    sorted_f1 = sorted(per_class_f1.items(), key=lambda item: item[1], reverse=True)
    best_f1_score = sorted_f1[0][1]
    lowest_f1_score = sorted_f1[-1][1]

    best_signs = [k for k, v in per_class_f1.items() if v == best_f1_score]
    weakest_signs = [k for k, v in per_class_f1.items() if v == lowest_f1_score]

    print(f"  ★ Best Performing Sign(s):    {', '.join(best_signs)} ({best_f1_score:.1f}% F1)")
    if best_f1_score != lowest_f1_score:
        print(f"  ⚠️ Lowest Performing Sign(s):  {', '.join(weakest_signs)} ({lowest_f1_score:.1f}% F1)")
    print()

    # 8. Generate & Save Confusion Matrix Plot
    print_banner("GENERATING ARTIFACTS")
    cm = confusion_matrix(y_test_enc, y_pred_enc)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix — {model_type} ({accuracy*100:.1f}% Accuracy)", fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    # Add numeric annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                format(val, "d"),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
                fontsize=11,
            )

    ax.set_ylabel("True Ground-Truth Sign", fontweight="bold", fontsize=11)
    ax.set_xlabel("Predicted Sign", fontweight="bold", fontsize=11)
    plt.tight_layout()

    fig.savefig(CONFUSION_MATRIX_PATH)
    plt.close(fig)
    print(f"  ✓ Confusion Matrix Image:      {CONFUSION_MATRIX_PATH.relative_to(BASE_DIR)}")

    # 9. Save Classification Report (JSON & CSV)
    with open(CLASSIFICATION_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"  ✓ Classification Report JSON:  {CLASSIFICATION_REPORT_JSON.relative_to(BASE_DIR)}")

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(CLASSIFICATION_REPORT_CSV, index=True)
    print(f"  ✓ Classification Report CSV:   {CLASSIFICATION_REPORT_CSV.relative_to(BASE_DIR)}")

    # 10. Save Evaluation Summary JSON
    summary = {
        "model_type": model_type,
        "n_classes": len(classes),
        "classes": classes,
        "n_features": expected_features,
        "total_dataset_samples": total_samples,
        "evaluation_samples": test_samples,
        "train_samples": train_samples,
        "split_parameters": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "metrics": {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
        },
        "per_class_f1": per_class_f1,
        "best_performing_signs": best_signs,
        "weakest_performing_signs": weakest_signs,
    }

    with open(EVALUATION_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Evaluation Summary JSON:     {EVALUATION_SUMMARY_JSON.relative_to(BASE_DIR)}")
    print()
    print_banner("EVALUATION COMPLETE — ALL ARTIFACTS GENERATED")
    return summary


if __name__ == "__main__":
    evaluate_model()
