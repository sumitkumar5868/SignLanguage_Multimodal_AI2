"""Phase 6: Machine Learning Model Training and Evaluation Pipeline.

Dataset: dataset/Forproject.csv
  - 1,610 samples
  - 126 landmark features  (h1: 63 + h2: 63)
  - 17 sign classes
  - Single-hand samples: h2 slot zero-padded (1,398 rows)
  - Two-hand samples: both h1 and h2 filled (212 rows)

Pipeline:
CSV Dataset → Load → Validate → Dry-Run Check →
Separate Features/Labels → Train/Test Split →
Feature Scaling → Model Training → Evaluation →
Save Model/Artifacts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT — h1/h2 Hand-Ordering Convention Notice
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The Forproject.csv dataset was collected using DETECTION-ORDER ordering:
  h1 = first hand detected by MediaPipe (regardless of left/right)
  h2 = second hand if present, else zeros

The live inference pipeline in landmark_features.py uses
LEFT/RIGHT DETERMINISTIC ordering:
  h1 = LEFT hand
  h2 = RIGHT hand

For single-hand signs (1,398/1,610 rows) this mismatch has NO effect
because h1 is always filled and h2 is always zero in both conventions.

For two-hand signs (212/1,610 rows — BATHROOM, EMERGENCY, HELP, REST, SICK)
the slot assignment during live inference may differ from how those rows were
recorded.  This creates a potential inconsistency for two-hand signs only.

ACTION REQUIRED BEFORE DEPLOYMENT:
  Option A — Re-collect two-hand samples using handedness-aware code
             (left/right deterministic).
  Option B — Update landmark_features.py to use detection-order during
             inference to match the training data.
  Option C — Accept the risk: for many two-hand signs both hands are
             symmetrical so the mismatch may not degrade accuracy.

This training code preserves the CSV exactly as-is. No reordering is
applied. The mismatch is documented here and in model_metadata.json.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "Forproject.csv"
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CONFUSION_MATRIX_PATH = DATA_DIR / "confusion_matrix.png"

# Model artifact paths
MODEL_PATH = MODEL_DIR / "sign_language_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Configuration — updated for 126-feature / 17-class dataset
# ---------------------------------------------------------------------------
EXPECTED_FEATURES = 126          # 63 (h1: 21 landmarks × 3) + 63 (h2: 21 landmarks × 3)
EXPECTED_TOTAL_COLUMNS = 127     # 126 features + 1 label column
LABEL_COLUMN = "label"

# Classes are read DYNAMICALLY from the dataset — not hard-coded.
# Old hard-coded list ["HELLO","I_LOVE_YOU","I_HATE_YOU","I_EAT","THANK_YOU"]
# has been intentionally removed.

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 200

# Expected feature column naming pattern for validation
# h1_x0 … h1_z20 followed by h2_x0 … h2_z20
_EXPECTED_FEATURE_COLUMNS = [
    f"{hand}_{coord}{idx}"
    for hand in ("h1", "h2")
    for idx in range(21)
    for coord in ("x", "y", "z")
]


def print_header():
    """Print the project header."""
    print("=" * 80)
    print(" SIGN LANGUAGE MULTIMODAL AI")
    print("=" * 80)
    print()
    print("PHASE 6: MACHINE LEARNING TRAINING")
    print()
    print("Dataset  :", DATASET_PATH.name)
    print("Features :", EXPECTED_FEATURES, "(h1: 63 + h2: 63)")
    print("Classes  : dynamic (read from dataset)")
    print()


def load_dataset(path=None):
    """Load the dataset from CSV file.

    Args:
        path: Path to CSV file. Defaults to DATASET_PATH.

    Returns:
        DataFrame containing the dataset, or None if loading fails.
    """
    target_path = Path(path) if path else DATASET_PATH

    print("Dataset:")
    print(f"  {target_path}")
    print()

    if not target_path.exists():
        print(f"✗ ERROR: Dataset file not found: {target_path}")
        return None

    try:
        df = pd.read_csv(target_path)
        return df
    except Exception as exc:
        print(f"✗ ERROR: Failed to load dataset: {str(exc)}")
        return None


def validate_dataset(df):
    """Validate dataset structure and content against 126-feature / 17-class schema.

    Checks performed:
      1. Dataset is non-empty.
      2. Exactly EXPECTED_TOTAL_COLUMNS (127) columns present.
      3. Last column is 'label'.
      4. Exactly EXPECTED_FEATURES (126) feature columns present.
      5. Feature column names match expected h1/h2 pattern exactly.
      6. All feature columns are numeric (float or int).
      7. No NaN values in feature columns.
      8. No infinite values in feature columns.
      9. At least 2 classes present.
     10. Reports class distribution and imbalance.

    Args:
        df: DataFrame to validate.

    Returns:
        Tuple (is_valid: bool, classes: list[str]) — classes is the sorted
        list of unique class labels found in the dataset.
    """
    print("-" * 80)
    print("DATASET VALIDATION")
    print("-" * 80)
    print()

    if df is None or df.empty:
        print("✗ Dataset is empty or None")
        return False, []

    print(f"✓ Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")

    # ── 1. Total column count ────────────────────────────────────────────────
    if len(df.columns) != EXPECTED_TOTAL_COLUMNS:
        print(
            f"✗ Expected {EXPECTED_TOTAL_COLUMNS} columns "
            f"(126 features + 1 label), got {len(df.columns)}"
        )
        return False, []
    print(f"✓ Column count: {len(df.columns)} (correct)")

    # ── 2. Label column present and last ────────────────────────────────────
    if LABEL_COLUMN not in df.columns:
        print(f"✗ Label column '{LABEL_COLUMN}' not found")
        return False, []
    if df.columns[-1] != LABEL_COLUMN:
        print(
            f"✗ Label column must be the LAST column; "
            f"found it at position {list(df.columns).index(LABEL_COLUMN)}"
        )
        return False, []
    print(f"✓ Label column: '{LABEL_COLUMN}' (last column, correct)")

    # ── 3. Feature columns ───────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
    if len(feature_cols) != EXPECTED_FEATURES:
        print(
            f"✗ Expected {EXPECTED_FEATURES} feature columns, "
            f"got {len(feature_cols)}"
        )
        return False, []
    print(f"✓ Feature columns: {len(feature_cols)} (correct)")

    # ── 4. Feature column naming / ordering ─────────────────────────────────
    if feature_cols != _EXPECTED_FEATURE_COLUMNS:
        mismatches = [
            (i, exp, act)
            for i, (exp, act) in enumerate(
                zip(_EXPECTED_FEATURE_COLUMNS, feature_cols)
            )
            if exp != act
        ]
        print(
            f"✗ Feature column names/order do not match expected h1/h2 pattern. "
            f"First {min(5, len(mismatches))} mismatches:"
        )
        for idx, exp, act in mismatches[:5]:
            print(f"    col {idx}: expected '{exp}', got '{act}'")
        return False, []
    print("✓ Feature naming: h1_x0…h1_z20, h2_x0…h2_z20 (correct)")

    # ── 5. Feature dtype ─────────────────────────────────────────────────────
    non_numeric = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        print(f"✗ Non-numeric feature columns: {non_numeric[:5]}")
        return False, []
    print("✓ All feature columns are numeric")

    # ── 6. NaN check ─────────────────────────────────────────────────────────
    nan_count = df[feature_cols].isnull().sum().sum()
    if nan_count > 0:
        nan_cols = df[feature_cols].isnull().sum()
        nan_cols = nan_cols[nan_cols > 0]
        print(f"✗ Found {nan_count} NaN values in feature columns:")
        print(f"    {nan_cols.to_dict()}")
        return False, []
    print("✓ No NaN values")

    # ── 7. Infinite value check ──────────────────────────────────────────────
    inf_count = np.isinf(df[feature_cols].values).sum()
    if inf_count > 0:
        print(f"✗ Found {inf_count} infinite values in feature columns")
        return False, []
    print("✓ No infinite values")

    # ── 8. Classes ───────────────────────────────────────────────────────────
    classes = sorted(df[LABEL_COLUMN].unique().tolist())
    n_classes = len(classes)
    if n_classes < 2:
        print(f"✗ Need at least 2 classes, found {n_classes}: {classes}")
        return False, []
    print(f"✓ Classes: {n_classes} detected dynamically")

    # ── 9. Class distribution ────────────────────────────────────────────────
    print()
    print("  Class distribution:")
    dist = df[LABEL_COLUMN].value_counts().sort_index()
    max_samples = dist.max()
    min_samples = dist.min()
    for label in classes:
        count = dist.get(label, 0)
        bar = "█" * int(count / max_samples * 20)
        flag = " ⚠ imbalanced" if count >= min_samples * 3 else ""
        print(f"    {label:15} {count:4d}  {bar}{flag}")
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float("inf")
    print()
    print(f"  Imbalance ratio : {imbalance_ratio:.1f}× (max/min samples)")
    print(f"  class_weight    : 'balanced' will be used to compensate")

    # ── 10. h1/h2 convention warning ─────────────────────────────────────────
    h2_cols = [c for c in feature_cols if c.startswith("h2_")]
    h2_zero_rows = (df[h2_cols] == 0).all(axis=1).sum()
    both_hand_rows = len(df) - h2_zero_rows
    print()
    print("  Hand-slot analysis:")
    print(f"    Single-hand rows (h2 = zeros) : {h2_zero_rows:,}")
    print(f"    Two-hand rows (h2 filled)      : {both_hand_rows:,}")
    print()
    print(
        "  ⚠ NOTE — h1/h2 Convention Mismatch (training vs inference):\n"
        "    The CSV uses detection-order (h1=first detected hand).\n"
        "    The live pipeline uses left/right ordering (h1=LEFT, h2=RIGHT).\n"
        "    Impact: single-hand signs — NONE (h1 always filled).\n"
        "    Impact: two-hand signs    — possible slot swap for some samples.\n"
        "    See module docstring for resolution options.\n"
        "    This training run proceeds with the CSV exactly as provided."
    )

    print()
    return True, classes


def dry_run_validation(df, classes):
    """Perform a full static dry-run of the training pipeline without fitting any model.

    Verifies:
      - Feature/label separation produces correct shapes.
      - Train/test stratified split would succeed with the current class distribution.
      - StandardScaler can be fitted on training features.
      - LabelEncoder encodes all 17 classes correctly.
      - No class has fewer than 2 samples (required for stratified split).

    Args:
        df: Validated DataFrame.
        classes: Sorted list of class labels from validate_dataset().

    Returns:
        True if all checks pass, False otherwise.
    """
    print("-" * 80)
    print("DRY-RUN VALIDATION (no model training)")
    print("-" * 80)
    print()

    try:
        # Feature / label separation
        feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
        X = df[feature_cols].values
        y = df[LABEL_COLUMN].values
        print(f"✓ Feature matrix shape : {X.shape}")
        print(f"✓ Label vector shape   : {y.shape}")

        # Check minimum samples per class for stratified split
        min_count = df[LABEL_COLUMN].value_counts().min()
        required_test_count = max(1, int(min_count * TEST_SIZE))
        if min_count < 2:
            print(
                f"✗ Smallest class has {min_count} sample(s). "
                f"Stratified split requires at least 2 per class."
            )
            return False
        print(
            f"✓ Smallest class has {min_count} samples — "
            f"stratified split feasible (test quota ≈ {required_test_count})"
        )

        # Label encoding dry-run
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(f"✓ LabelEncoder: {len(le.classes_)} classes encoded correctly")

        # Train/test split dry-run
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_encoded,
        )
        print(f"✓ Train/test split: {len(X_train)} train / {len(X_test)} test")

        # Scaler dry-run
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(
            f"✓ StandardScaler fitted: "
            f"mean range [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]"
        )
        print(f"✓ Feature matrix after scaling shape: {X_train_scaled.shape}")
        print(
            f"✓ RandomForestClassifier(n_estimators={N_ESTIMATORS}, "
            f"class_weight='balanced') — configuration valid (not fitted)"
        )

        print()
        print(f"  Dry-run summary:")
        print(f"    Features        : {EXPECTED_FEATURES}")
        print(f"    Classes         : {len(classes)}")
        print(f"    Training rows   : {len(X_train)}")
        print(f"    Test rows       : {len(X_test)}")
        print(f"    Scaler          : StandardScaler")
        print(f"    Class weighting : balanced")
        print()
        print("✓ DRY-RUN PASSED — all pipeline stages validated successfully")
        print()
        return True

    except Exception as exc:
        print(f"✗ Dry-run failed: {exc}")
        return False


def separate_features_and_labels(df):
    """Separate features and labels from dataset.

    Args:
        df: DataFrame containing features and label.

    Returns:
        Tuple (X, y) containing features and labels, or (None, None) on error.
    """
    print("-" * 80)
    print("FEATURE AND LABEL SEPARATION")
    print("-" * 80)
    print()

    try:
        feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
        X = df[feature_cols].values
        y = df[LABEL_COLUMN].values

        print(f"Feature shape : {X.shape}")
        print(f"Label shape   : {y.shape}")
        print()

        return X, y
    except Exception as exc:
        print(f"✗ ERROR: Failed to separate features and labels: {str(exc)}")
        return None, None


def split_train_test(X, y_encoded, classes):
    """Split dataset into training and testing sets using stratification.

    Args:
        X: Features array.
        y_encoded: Encoded integer labels array.
        classes: Sorted list of class label strings (from LabelEncoder).

    Returns:
        Tuple (X_train, X_test, y_train, y_test) or (None, None, None, None) on error.
    """
    print("-" * 80)
    print("TRAIN / TEST SPLIT")
    print("-" * 80)
    print()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_encoded,
        )

        print(f"Training samples : {len(X_train)}")
        print(f"Testing samples  : {len(X_test)}")
        print()

        # Dynamic class distribution — uses actual classes, not hard-coded list
        print("Class distribution (train / test):")
        for idx, label in enumerate(classes):
            train_count = int(np.sum(y_train == idx))
            test_count = int(np.sum(y_test == idx))
            print(f"  {label:15}  Train: {train_count:3d}  Test: {test_count:3d}")
        print()

        return X_train, X_test, y_train, y_test
    except Exception as exc:
        print(f"✗ ERROR: Failed to split data: {str(exc)}")
        return None, None, None, None


def scale_features(X_train, X_test):
    """Scale features using StandardScaler fitted on training data only.

    Args:
        X_train: Training features.
        X_test: Testing features.

    Returns:
        Tuple (scaler, X_train_scaled, X_test_scaled) or (None, None, None) on error.
    """
    print("-" * 80)
    print("FEATURE SCALING")
    print("-" * 80)
    print()

    try:
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"✓ StandardScaler fitted on {len(X_train)} training samples")
        print(f"✓ Input features : {scaler.n_features_in_}")
        print(
            f"✓ Feature mean range  : [{scaler.mean_.min():.4f}, "
            f"{scaler.mean_.max():.4f}]"
        )
        print(
            f"✓ Feature scale range : [{scaler.scale_.min():.4f}, "
            f"{scaler.scale_.max():.4f}]"
        )
        print()

        return scaler, X_train_scaled, X_test_scaled
    except Exception as exc:
        print(f"✗ ERROR: Failed to scale features: {str(exc)}")
        return None, None, None


def train_model(X_train_scaled, y_train):
    """Train RandomForestClassifier with balanced class weights.

    class_weight='balanced' is required because Forproject.csv has a 6×
    imbalance (HELLO/YES/GOODBYE at 300 samples vs others at 50 samples).

    Args:
        X_train_scaled: Scaled training features (shape: N × 126).
        y_train: Encoded integer training labels.

    Returns:
        Trained model or None on error.
    """
    print("-" * 80)
    print("MODEL TRAINING")
    print("-" * 80)
    print()

    try:
        print(f"  Algorithm       : RandomForestClassifier")
        print(f"  Estimators      : {N_ESTIMATORS}")
        print(f"  Features        : {X_train_scaled.shape[1]}")
        print(f"  Training rows   : {X_train_scaled.shape[0]}")
        print(f"  Class weighting : balanced")
        print(f"  Random state    : {RANDOM_STATE}")
        print()
        print("Training...")

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",   # compensates for 6× imbalance
        )

        model.fit(X_train_scaled, y_train)

        print("✓ Model training completed")
        print()

        return model
    except Exception as exc:
        print(f"✗ ERROR: Failed to train model: {str(exc)}")
        return None


def evaluate_model(model, X_test_scaled, y_test, le):
    """Evaluate model on test set and print a full classification report.

    Args:
        model: Trained model.
        X_test_scaled: Scaled test features.
        y_test: Encoded integer test labels.
        le: Fitted LabelEncoder for decoding labels.

    Returns:
        Tuple (accuracy, y_pred, y_pred_labels, y_test_labels)
        or (None, None, None, None) on error.
    """
    print("-" * 80)
    print("MODEL EVALUATION")
    print("-" * 80)
    print()

    try:
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print()

        # Decode labels for human-readable classification report
        y_test_labels = le.inverse_transform(y_test)
        y_pred_labels = le.inverse_transform(y_pred)

        print("Classification Report:")
        print()
        print(classification_report(y_test_labels, y_pred_labels))

        return accuracy, y_pred, y_pred_labels, y_test_labels
    except Exception as exc:
        print(f"✗ ERROR: Failed to evaluate model: {str(exc)}")
        return None, None, None, None


def generate_confusion_matrix(y_test, y_pred, le):
    """Generate and save a fully dynamic confusion matrix.

    The grid dimensions are determined automatically from the number of
    classes in the label encoder — no hard-coded 5-class assumption.

    Args:
        y_test: True encoded test labels.
        y_pred: Predicted encoded labels.
        le: Fitted LabelEncoder providing class names.

    Returns:
        True on success, False on error.
    """
    print("-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)
    print()

    try:
        # Dynamic class list from label encoder — NOT hard-coded
        class_labels = list(le.classes_)
        n_classes = len(class_labels)

        cm = confusion_matrix(y_test, y_pred)

        # Scale figure size dynamically: ~0.7 inches per class, min 8×6
        fig_size = max(8, int(n_classes * 0.75))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        # Dynamic tick marks and labels
        tick_marks = np.arange(n_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_labels, fontsize=9)

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title(
            f"Confusion Matrix — Sign Language Classification "
            f"({n_classes} classes)"
        )

        # Annotate cells
        threshold = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > threshold else "black",
                )

        plt.tight_layout()

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(CONFUSION_MATRIX_PATH, dpi=100, bbox_inches="tight")
        print(f"✓ Confusion matrix saved: {CONFUSION_MATRIX_PATH}")
        print()

        plt.close(fig)
        return True
    except Exception as exc:
        print(f"✗ ERROR: Failed to generate confusion matrix: {str(exc)}")
        return False


def save_model_artifacts(model, scaler, le, df, X_train, y_train, accuracy):
    """Save trained model, scaler, label encoder, and rich metadata.

    Metadata includes:
      - feature_count, class_count, classes list
      - dataset_path, sample_count
      - class distribution
      - preprocessing info (scaler, ordering, weighting)
      - h1/h2 convention notice

    Args:
        model: Trained RandomForestClassifier.
        scaler: Fitted StandardScaler.
        le: Fitted LabelEncoder.
        df: Original DataFrame (for class distribution).
        X_train: Training features array (for sample counts).
        y_train: Encoded training labels.
        accuracy: Float accuracy on test set.

    Returns:
        True on success, False on error.
    """
    print("-" * 80)
    print("SAVING MODEL FILES")
    print("-" * 80)
    print()

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, MODEL_PATH)
        print(f"✓ {MODEL_PATH.name}")

        joblib.dump(scaler, SCALER_PATH)
        print(f"✓ {SCALER_PATH.name}")

        joblib.dump(le, LABEL_ENCODER_PATH)
        print(f"✓ {LABEL_ENCODER_PATH.name}")

        # Dynamic class distribution
        dist = df[LABEL_COLUMN].value_counts().sort_index()
        class_distribution = {
            str(label): int(count) for label, count in dist.items()
        }

        # Two-hand row count for metadata
        h2_cols = [c for c in df.columns if c.startswith("h2_")]
        h2_zero_rows = int((df[h2_cols] == 0).all(axis=1).sum())
        two_hand_rows = len(df) - h2_zero_rows

        metadata = {
            # ── Core model info ──────────────────────────────────────────────
            "model": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",

            # ── Feature configuration ────────────────────────────────────────
            "feature_count": int(scaler.n_features_in_),
            "feature_layout": {
                "h1": "landmarks 0–20 × (x, y, z) = 63 features",
                "h2": "landmarks 0–20 × (x, y, z) = 63 features",
                "total": 126,
            },

            # ── Class configuration ──────────────────────────────────────────
            "class_count": int(len(le.classes_)),
            "classes": [str(c) for c in le.classes_],
            "class_distribution": class_distribution,

            # ── Dataset info ─────────────────────────────────────────────────
            "dataset_path": str(DATASET_PATH.name),
            "dataset_samples": len(df),
            "training_samples": len(X_train),
            "testing_samples": len(df) - len(X_train),
            "single_hand_rows": h2_zero_rows,
            "two_hand_rows": two_hand_rows,

            # ── Preprocessing ────────────────────────────────────────────────
            "scaler": "StandardScaler (fitted on training data only)",
            "label_encoder": "LabelEncoder",

            # ── Accuracy ─────────────────────────────────────────────────────
            "accuracy": float(accuracy),

            # ── h1/h2 convention notice ──────────────────────────────────────
            "h1_h2_convention_notice": (
                "Dataset collected in detection-order (h1=first detected hand). "
                "Live inference pipeline uses left/right ordering (h1=LEFT, h2=RIGHT). "
                "Impact is negligible for single-hand signs. "
                "Two-hand signs may have slot inconsistency. "
                "See modules/train_model.py docstring for resolution options."
            ),
        }

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ {METADATA_PATH.name}")

        if CONFUSION_MATRIX_PATH.exists():
            print(f"✓ confusion_matrix.png")

        print()
        return True
    except Exception as exc:
        print(f"✗ ERROR: Failed to save model artifacts: {str(exc)}")
        return False


def test_model_loading():
    """Test that saved model and artifacts can be loaded correctly.

    Returns:
        True if all files load successfully, False otherwise.
    """
    print("-" * 80)
    print("MODEL LOADING TEST")
    print("-" * 80)
    print()

    try:
        model = joblib.load(MODEL_PATH)
        print(
            f"✓ Model loaded — n_features_in_: {model.n_features_in_}, "
            f"n_classes_: {model.n_classes_}"
        )

        scaler = joblib.load(SCALER_PATH)
        print(f"✓ Scaler loaded — n_features_in_: {scaler.n_features_in_}")

        le = joblib.load(LABEL_ENCODER_PATH)
        print(f"✓ LabelEncoder loaded — {len(le.classes_)} classes: {list(le.classes_)}")

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(
            f"✓ Metadata loaded — "
            f"features: {metadata.get('feature_count')}, "
            f"classes: {metadata.get('class_count')}, "
            f"accuracy: {metadata.get('accuracy', 0):.4f}"
        )

        print()
        return True
    except Exception as exc:
        print(f"✗ ERROR: Failed to load model artifacts: {str(exc)}")
        return False


def main():
    """Main training pipeline.

    Steps:
      1. Print header
      2. Load dataset (Forproject.csv)
      3. Validate dataset (126 features, 127 columns, dynamic classes)
      4. Dry-run validation (no model fitting)
      5. Separate features / labels
      6. Encode labels (LabelEncoder)
      7. Train/test stratified split
      8. Scale features (StandardScaler)
      9. Train RandomForestClassifier (class_weight='balanced')
     10. Evaluate on test set
     11. Generate dynamic confusion matrix (17×17)
     12. Save model artifacts + rich metadata
     13. Verify saved artifacts can be reloaded
    """
    print_header()

    # Step 1 — Load
    df = load_dataset()
    if df is None:
        return 1

    # Step 2 — Validate (returns dynamic class list)
    is_valid, classes = validate_dataset(df)
    if not is_valid:
        return 1

    # Step 3 — Dry-run (validates pipeline shapes without fitting RandomForest)
    if not dry_run_validation(df, classes):
        return 1

    # Step 4 — Separate
    X, y = separate_features_and_labels(df)
    if X is None or y is None:
        return 1

    # Step 5 — Encode labels dynamically
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # le.classes_ == sorted unique labels from dataset (17 classes)

    # Step 6 — Split
    X_train, X_test, y_train, y_test = split_train_test(X, y_encoded, list(le.classes_))
    if X_train is None:
        return 1

    # Step 7 — Scale
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    if scaler is None:
        return 1

    # Step 8 — Train
    model = train_model(X_train_scaled, y_train)
    if model is None:
        return 1

    # Step 9 — Evaluate
    accuracy, y_pred, y_pred_labels, y_test_labels = evaluate_model(
        model, X_test_scaled, y_test, le
    )
    if accuracy is None:
        return 1

    # Step 10 — Confusion matrix (dynamic, 17×17)
    if not generate_confusion_matrix(y_test, y_pred, le):
        return 1

    # Step 11 — Save artifacts
    if not save_model_artifacts(model, scaler, le, df, X_train, y_train, accuracy):
        return 1

    # Step 12 — Verify loading
    if not test_model_loading():
        return 1

    print("=" * 80)
    print(" PHASE 6 COMPLETED")
    print(f" Dataset   : {DATASET_PATH.name}")
    print(f" Features  : {EXPECTED_FEATURES}")
    print(f" Classes   : {len(le.classes_)}")
    print(f" Accuracy  : {accuracy * 100:.2f}%")
    print("=" * 80)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
