"""
Auto-Train Script — After collecting my_fresh_dataset.csv
Trains a new wrist-relative model and integrates it into the web app.
Run this after: python collect_my_data.py
"""

import json, sys, hashlib, time
from pathlib import Path

ROOT = Path("/Users/rohitroy/Downloads/SignLanguage_Multimodal_AI")
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATASET_IN  = ROOT / "dataset" / "my_fresh_dataset.csv"
MODEL_DIR   = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH   = MODEL_DIR / "sign_language_model_mydata.pkl"
SCALER_PATH  = MODEL_DIR / "scaler_mydata.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder_mydata.pkl"
META_PATH    = MODEL_DIR / "model_metadata_mydata.json"
CM_PATH      = MODEL_DIR / "confusion_matrix_mydata.png"

SIGNS_ORDER = [
    "HELLO", "GOODBYE", "YES", "NO", "PLEASE", "THANK_YOU", "SORRY",
    "HELP", "WATER", "FOOD", "BATHROOM", "PAIN", "SICK", "SLEEP",
    "REST", "STOP", "EMERGENCY"
]
RANDOM_STATE = 42
MIN_SAMPLES  = 30   # minimum per class to include

print("=" * 60)
print("  TRAIN MODEL ON MY FRESH DATASET")
print("=" * 60)

# Load
if not DATASET_IN.exists():
    print(f"ERROR: {DATASET_IN} not found. Run collect_my_data.py first.")
    sys.exit(1)

df = pd.read_csv(DATASET_IN)
label_col    = "label"
feature_cols = [c for c in df.columns if c != label_col]

print(f"\nLoaded: {len(df)} rows, {len(feature_cols)} features")
print("\nSamples per class:")
vc = df[label_col].value_counts().sort_index()
print(vc.to_string())

# Filter classes with too few samples
valid_classes = [c for c in SIGNS_ORDER if vc.get(c, 0) >= MIN_SAMPLES]
skipped = [c for c in SIGNS_ORDER if vc.get(c, 0) < MIN_SAMPLES]
if skipped:
    print(f"\nWARNING: Skipping {skipped} — less than {MIN_SAMPLES} samples")

df = df[df[label_col].isin(valid_classes)]
print(f"\nUsing {len(df)} rows, {len(valid_classes)} classes")

# Scale normalization function for distance invariance
def normalize_scale_63(feats_63):
    feats = np.array(feats_63, dtype=float).copy()
    if np.all(feats == 0):
        return feats
    mx, my, mz = feats[27], feats[28], feats[29]
    scale = (mx * mx + my * my + mz * mz) ** 0.5
    if scale > 1e-4:
        feats /= scale
    return feats

h1_cols = [c for c in feature_cols if c.startswith('h1_')]
h2_cols = [c for c in feature_cols if c.startswith('h2_')]

print("Applying Distance-Invariant Palm Scale Normalization ...")
norm_rows = []
for _, row in df.iterrows():
    h1 = normalize_scale_63(row[h1_cols].values)
    h2 = normalize_scale_63(row[h2_cols].values)
    norm_rows.append(list(h1) + list(h2) + [row[label_col]])

df = pd.DataFrame(norm_rows, columns=df.columns)

# Prepare
encoder     = LabelEncoder()
y_all_enc   = encoder.fit_transform(df[label_col].values)
class_names = list(encoder.classes_)

# Split first so test set remains pure real unaugmented data
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all_enc
)
print(f"Original Split -> Train: {len(train_df)}  |  Test: {len(test_df)}")

# ── Data Augmentation for Robust Real-World Detection ───────────────────────
print("\nApplying Multi-Angle & Hand-Swap Data Augmentation ...")

def rotate_hand_63(feats_63, angle_deg):
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    out = np.array(feats_63, dtype=float).copy()
    for i in range(21):
        x = out[i*3]
        y = out[i*3 + 1]
        out[i*3]     = x * c - y * s
        out[i*3 + 1] = x * s + y * c
    return out

aug_rows = []

for _, row in train_df.iterrows():
    lbl   = row[label_col]
    h1    = row[h1_cols].values
    h2    = row[h2_cols].values
    is_2h = (row['h2_x1'] != 0)

    # 1. Hand swap invariance for ALL two-hand signs (Left/Right flip immunity)
    if is_2h:
        aug_rows.append(list(h2) + list(h1) + [lbl])

    # 2. EMERGENCY specific: Wide X-shape arm cross angles (outward tilt +-20 to +-45 deg)
    if lbl == "EMERGENCY":
        for a1, a2 in [(25, -25), (35, -35), (45, -45), (-25, 25), (-35, 35), (-45, 45)]:
            h1_r = rotate_hand_63(h1, a1)
            h2_r = rotate_hand_63(h2, a2)
            aug_rows.append(list(h1_r) + list(h2_r) + [lbl])
            aug_rows.append(list(h2_r) + list(h1_r) + [lbl])  # also swapped

    # 3. Multi-angle rotations for ALL signs (+-15, +-25 deg)
    for ang in [-25, -15, 15, 25]:
        h1_r = rotate_hand_63(h1, ang)
        h2_r = rotate_hand_63(h2, ang) if is_2h else h2
        aug_rows.append(list(h1_r) + list(h2_r) + [lbl])
        if is_2h:
            aug_rows.append(list(h2_r) + list(h1_r) + [lbl])

df_aug = pd.DataFrame(aug_rows, columns=df.columns)
train_df_augmented = pd.concat([train_df, df_aug], ignore_index=True)
print(f"Augmented Train: {len(train_df_augmented)} rows (from {len(train_df)})")

X_train = train_df_augmented[feature_cols].values
y_train = encoder.transform(train_df_augmented[label_col].values)

X_test  = test_df[feature_cols].values
y_test  = encoder.transform(test_df[label_col].values)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Train
print("\nTraining RandomForest (300 estimators) on augmented dataset ...")
t0  = time.time()
clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                              n_jobs=-1, min_samples_leaf=2)
clf.fit(X_train_sc, y_train)
print(f"Done in {time.time()-t0:.1f}s")

# Evaluate
y_pred      = clf.predict(X_test_sc)
acc         = accuracy_score(y_test, y_pred)
w_f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
w_prec      = precision_score(y_test, y_pred, average="weighted", zero_division=0)
macro_prec  = precision_score(y_test, y_pred, average="macro",    zero_division=0)
macro_rec   = recall_score(y_test, y_pred,    average="macro",    zero_division=0)

print(f"\n{'='*40}")
print(f"  Accuracy    : {acc*100:.2f}%")
print(f"  Weighted F1 : {w_f1*100:.2f}%")
print(f"  Macro F1    : {macro_f1*100:.2f}%")
print(f"{'='*40}")
print()
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# Confusion matrix
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
im  = ax.imshow(cm, cmap=plt.cm.Blues)
fig.colorbar(im, ax=ax)
ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
       xticklabels=class_names, yticklabels=class_names,
       xlabel="Predicted", ylabel="True",
       title=f"My Personal Model — Accuracy: {acc*100:.2f}%")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.savefig(CM_PATH, dpi=150)
plt.close(fig)
print(f"Confusion matrix saved: {CM_PATH}")

# Save artifacts
joblib.dump(clf,     MODEL_PATH)
joblib.dump(scaler,  SCALER_PATH)
joblib.dump(encoder, ENCODER_PATH)

report_dict = classification_report(y_test, y_pred, target_names=class_names,
                                     zero_division=0, output_dict=True)
per_class = {c: {k: report_dict[c][k] for k in ("precision","recall","f1-score","support")}
             for c in class_names}

metadata = {
    "model_name":        "sign_language_model_mydata",
    "feature_type":      "wrist_relative",
    "dataset_source":    str(DATASET_IN),
    "algorithm":         "RandomForestClassifier",
    "num_features":      len(feature_cols),
    "num_classes":       len(class_names),
    "classes":           class_names,
    "train_samples":     int(len(X_train)),
    "test_samples":      int(len(X_test)),
    "accuracy":          float(acc),
    "weighted_f1":       float(w_f1),
    "macro_f1":          float(macro_f1),
    "per_class_metrics": per_class,
}
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nArtifacts saved:")
print(f"  {MODEL_PATH}")
print(f"  {SCALER_PATH}")
print(f"  {ENCODER_PATH}")
print(f"  {META_PATH}")

# Update predictor.py
PREDICTOR = ROOT / "modules" / "predictor.py"
text = PREDICTOR.read_text()
text = text.replace(
    '"sign_language_model_wristrel_17class.pkl"',
    '"sign_language_model_mydata.pkl"'
).replace(
    '"scaler_wristrel_17class.pkl"',
    '"scaler_mydata.pkl"'
).replace(
    '"label_encoder_wristrel_17class.pkl"',
    '"label_encoder_mydata.pkl"'
).replace(
    '"model_metadata_wristrel_17class.json"',
    '"model_metadata_mydata.json"'
).replace(
    '"Wrist-Relative 17-Class Model"',
    f'"My Personal {len(class_names)}-Class Model"'
)
PREDICTOR.write_text(text)
print(f"\npredictor.py updated to use new model.")

# Update app.py
APP = ROOT / "app.py"
app_text = APP.read_text()
app_text = app_text.replace(
    '"Wrist-Relative 17-Class Model"',
    f'"My Personal {len(class_names)}-Class Model"'
).replace(
    '"sign_language_model_wristrel_17class.pkl"',
    '"sign_language_model_mydata.pkl"'
)
APP.write_text(app_text)
print("app.py updated.")

print("\n" + "="*60)
print(f"  DONE! Model trained on YOUR personal data.")
print(f"  Accuracy: {acc*100:.2f}%  |  Classes: {len(class_names)}")
print(f"\n  Now restart the web app:")
print(f"  venv/bin/python3 app.py --port 5001")
print("="*60)
