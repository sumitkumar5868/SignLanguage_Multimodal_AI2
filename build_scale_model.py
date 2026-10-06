"""
STEP 20 — Scale-Normalized Wrist-Relative Model Pipeline
=========================================================
Input  : dataset/my_fresh_dataset.csv   (already wrist-relative, h1/h2 slots, 126 cols + label)
Outputs: dataset/Forproject_wristrel_scale.csv
         dataset/Forproject_wristrel_scale_balanced.csv
         model/sign_language_model_wristrel_scale_17class.pkl
         model/scaler_wristrel_scale_17class.pkl
         model/label_encoder_wristrel_scale_17class.pkl
         model/model_metadata_wristrel_scale_17class.json
         model/confusion_matrix_wristrel_scale_17class.png

Scale formula (IDENTICAL in training AND live inference):
    input: wrist-relative features where wrist = (0,0,0)
    scale = L2 norm of landmark 9 (Middle MCP) wrist-relative coords
          = sqrt(f[27]^2 + f[28]^2 + f[29]^2)
    normalized[i] = wristrel[i] / scale
    Each hand is normalized independently using its own scale.
    Zero-padded h2 block stays all zeros.
"""

import json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT        = Path("/Users/rohitroy/Downloads/SignLanguage_Multimodal_AI")
DATASET_IN  = ROOT / "dataset" / "my_fresh_dataset.csv"
DS_SCALE    = ROOT / "dataset" / "Forproject_wristrel_scale.csv"
DS_BAL      = ROOT / "dataset" / "Forproject_wristrel_scale_balanced.csv"
MODEL_DIR   = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH   = MODEL_DIR / "sign_language_model_wristrel_scale_17class.pkl"
SCALER_PATH  = MODEL_DIR / "scaler_wristrel_scale_17class.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder_wristrel_scale_17class.pkl"
META_PATH    = MODEL_DIR / "model_metadata_wristrel_scale_17class.json"
CM_PATH      = MODEL_DIR / "confusion_matrix_wristrel_scale_17class.png"

SIGNS_ORDER = [
    "HELLO", "GOODBYE", "YES", "NO", "PLEASE", "THANK_YOU", "SORRY",
    "HELP", "WATER", "FOOD", "BATHROOM", "PAIN", "SICK", "SLEEP",
    "REST", "STOP", "EMERGENCY"
]
BALANCE_TARGET = 800
RANDOM_STATE   = 42

# ===========================================================================
# SCALE NORMALIZATION FORMULA  (must match _make_wristrel_63 exactly)
# ===========================================================================

def scale_normalize_63(v63):
    """
    Normalize a wrist-relative 63-float block by the L2 norm of landmark 9.
    Identical formula to modules/landmark_features.py::_make_wristrel_63.
    """
    v = np.array(v63, dtype=float)
    if np.all(v == 0.0):
        return v   # zero-padded h2 slot — leave untouched
    scale = float(np.linalg.norm(v[27:30]))   # landmark 9 wrist-relative L2
    if scale < 1e-4:
        return v
    return v / scale


def scale_normalize_126(row126):
    h1 = scale_normalize_63(row126[:63])
    h2 = scale_normalize_63(row126[63:])
    return np.concatenate([h1, h2])


# ===========================================================================
# 1. Load dataset and apply scale normalization
# ===========================================================================

print("=" * 65)
print("  STEP 20 — Scale-Normalized Wrist-Relative Model Pipeline")
print("=" * 65)

if not DATASET_IN.exists():
    print(f"ERROR: {DATASET_IN} not found."); sys.exit(1)

df = pd.read_csv(DATASET_IN)
label_col    = "label"
feature_cols = [c for c in df.columns if c != label_col]
assert len(feature_cols) == 126, f"Expected 126 features, got {len(feature_cols)}"

print(f"\nLoaded: {len(df)} rows x {len(feature_cols)} features")
print("Samples per class:")
print(df[label_col].value_counts().sort_index().to_string())

print("\n[1/6] Applying palm-scale normalization ...")
X_raw    = df[feature_cols].values.astype(float)
X_scaled = np.array([scale_normalize_126(r) for r in X_raw])
df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
df_scaled[label_col] = df[label_col].values
df_scaled.to_csv(DS_SCALE, index=False)
print(f"      Saved: {DS_SCALE}  ({len(df_scaled)} rows)")

# ===========================================================================
# 2. Balance dataset
# ===========================================================================

print(f"\n[2/6] Creating balanced dataset ({BALANCE_TARGET} samples/class) ...")
rng = np.random.default_rng(RANDOM_STATE)
balanced_parts = []
for cls in SIGNS_ORDER:
    cls_df = df_scaled[df_scaled[label_col] == cls]
    n = len(cls_df)
    if n == 0:
        print(f"      WARN: {cls} has 0 samples — skip"); continue
    if n >= BALANCE_TARGET:
        sampled = cls_df.sample(n=BALANCE_TARGET, random_state=RANDOM_STATE)
    else:
        extra   = cls_df.sample(n=BALANCE_TARGET - n, replace=True, random_state=RANDOM_STATE)
        sampled = pd.concat([cls_df, extra], ignore_index=True)
    balanced_parts.append(sampled)
    print(f"      {cls:12s}: {n:5d} -> {len(sampled)}")

df_bal = pd.concat(balanced_parts, ignore_index=True).sample(
    frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
df_bal.to_csv(DS_BAL, index=False)
print(f"\n      Saved: {DS_BAL}  ({len(df_bal)} rows)")

# ===========================================================================
# 3. Train/test split + augmentation
# ===========================================================================

print("\n[3/6] Train/test split + augmentation ...")
encoder = LabelEncoder()
encoder.fit([c for c in SIGNS_ORDER if c in df_bal[label_col].unique()])
class_names = list(encoder.classes_)

y_all = encoder.transform(df_bal[label_col].values)
train_df, test_df = train_test_split(
    df_bal, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all)
print(f"      Train: {len(train_df)}  |  Test: {len(test_df)}")

def rotate_63(feats, angle_deg):
    out = np.array(feats, dtype=float)
    r = np.radians(angle_deg); c, s = np.cos(r), np.sin(r)
    for i in range(21):
        x, y = out[i*3], out[i*3+1]
        out[i*3], out[i*3+1] = x*c - y*s, x*s + y*c
    return out

h1_cols = [c for c in feature_cols if c.startswith('h1_')]
h2_cols = [c for c in feature_cols if c.startswith('h2_')]

aug_rows = []
for _, row in train_df.iterrows():
    lbl  = row[label_col]
    h1   = row[h1_cols].values.astype(float)
    h2   = row[h2_cols].values.astype(float)
    is2h = not np.all(h2 == 0.0)
    if is2h:
        aug_rows.append(list(h2) + list(h1) + [lbl])
    for ang in [-25, -15, 15, 25]:
        h1r = rotate_63(h1, ang)
        h2r = rotate_63(h2, ang) if is2h else h2
        aug_rows.append(list(h1r) + list(h2r) + [lbl])
        if is2h: aug_rows.append(list(h2r) + list(h1r) + [lbl])
    if lbl == "EMERGENCY":
        for a1, a2 in [(25,-25),(35,-35),(45,-45)]:
            h1r, h2r = rotate_63(h1,a1), rotate_63(h2,a2)
            aug_rows.append(list(h1r)+list(h2r)+[lbl])
            aug_rows.append(list(h2r)+list(h1r)+[lbl])

df_aug    = pd.DataFrame(aug_rows, columns=list(feature_cols)+[label_col])
train_aug = pd.concat([train_df, df_aug], ignore_index=True)
print(f"      Augmented train: {len(train_aug)} rows")

X_train = train_aug[feature_cols].values
y_train = encoder.transform(train_aug[label_col].values)
X_test  = test_df[feature_cols].values
y_test  = encoder.transform(test_df[label_col].values)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ===========================================================================
# 4. Train
# ===========================================================================

print("\n[4/6] Training RandomForest (400 estimators) ...")
t0  = time.time()
clf = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE,
                              n_jobs=-1, min_samples_leaf=2, max_features="sqrt")
clf.fit(X_train_sc, y_train)
print(f"      Done in {time.time()-t0:.1f}s")

# ===========================================================================
# 5. Evaluate + scale invariance test
# ===========================================================================

print("\n[5/6] Evaluating ...")
y_pred   = clf.predict(X_test_sc)
acc      = accuracy_score(y_test, y_pred)
w_f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average="macro",    zero_division=0)

print(f"\n{'='*55}")
print(f"  Accuracy    : {acc*100:.2f}%")
print(f"  Weighted F1 : {w_f1*100:.2f}%")
print(f"  Macro F1    : {macro_f1*100:.2f}%")
print(f"{'='*55}")
print()
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# Scale-invariance simulation
print("--- Scale-Invariance Test (simulated distances) ---")
print("  Multiply wristrel features by fake scale -> re-normalize -> predict")
print()

test_sign_names = [s for s in ["HELLO","GOODBYE","YES","NO","STOP","SORRY","WATER"] if s in class_names]

def pred_at_scale(norm_row, fake_scale):
    fake_wr  = norm_row * fake_scale           # simulate different distance
    renorm   = scale_normalize_126(fake_wr)    # re-apply same normalization
    xs       = scaler.transform(renorm.reshape(1,-1))
    return encoder.inverse_transform(clf.predict(xs))[0]

acc_by_scale = {0.5: [], 1.0: [], 2.0: []}
for sname in test_sign_names:
    idx  = class_names.index(sname)
    mask = (y_test == idx)
    if not mask.any(): continue
    row  = X_test[mask][0]
    preds = {sc: pred_at_scale(row, sc) for sc in [0.5, 1.0, 2.0]}
    for sc, p in preds.items(): acc_by_scale[sc].append(p == sname)
    consistent = "OK" if len(set(preds.values())) == 1 else f"VARIES {preds}"
    print(f"  {sname:12s}  close={preds[2.0]:12s} normal={preds[1.0]:12s} far={preds[0.5]:12s}  [{consistent}]")

print()
for sc, correct in acc_by_scale.items():
    label = {0.5:"far",1.0:"normal",2.0:"close"}[sc]
    a = sum(correct)/len(correct)*100 if correct else 0
    print(f"  {label:6s} distance: {a:.0f}% ({sum(correct)}/{len(correct)})")

# ===========================================================================
# 6. Save
# ===========================================================================

print("\n[6/6] Saving artifacts ...")
joblib.dump(clf,     MODEL_PATH);  print(f"      {MODEL_PATH}")
joblib.dump(scaler,  SCALER_PATH); print(f"      {SCALER_PATH}")
joblib.dump(encoder, ENCODER_PATH);print(f"      {ENCODER_PATH}")

cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
im  = ax.imshow(cm, cmap=plt.cm.Blues)
fig.colorbar(im, ax=ax)
ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
       xticklabels=class_names, yticklabels=class_names,
       xlabel="Predicted", ylabel="True",
       title=f"Scale-Normalized Model — Accuracy: {acc*100:.2f}%")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i,j]>thresh else "black")
fig.tight_layout()
plt.savefig(CM_PATH, dpi=150)
plt.close(fig)
print(f"      {CM_PATH}")

report_dict = classification_report(y_test, y_pred, target_names=class_names,
                                     zero_division=0, output_dict=True)
per_class = {c:{k:report_dict[c][k] for k in ("precision","recall","f1-score","support")}
             for c in class_names}
metadata = {
    "model_name":  "sign_language_model_wristrel_scale_17class",
    "feature_type":"wrist_relative_scale_normalized",
    "scale_reference": "landmark_9_Middle_MCP_wristrel_L2_norm",
    "scale_formula":   "normalized[i] = wristrel[i] / ||wristrel_lm9||",
    "dataset_source":  str(DATASET_IN),
    "algorithm":       "RandomForestClassifier",
    "n_estimators":    400,
    "num_features":    126,
    "num_classes":     len(class_names),
    "classes":         class_names,
    "train_samples":   int(len(X_train)),
    "test_samples":    int(len(X_test)),
    "accuracy":        float(acc),
    "weighted_f1":     float(w_f1),
    "macro_f1":        float(macro_f1),
    "balance_target":  BALANCE_TARGET,
    "per_class_metrics": per_class,
    "live_inference_uses_identical_transform": True,
}
with open(META_PATH,"w") as f: json.dump(metadata, f, indent=2)
print(f"      {META_PATH}")

print("\n" + "="*65)
print(f"  DONE. Accuracy: {acc*100:.2f}%  Macro F1: {macro_f1*100:.2f}%")
print(f"  Features: 126 | Scale ref: lm9 wristrel L2 norm")
print("="*65)
print()
print("  Now run:  python switch_to_scale_model.py")
print("="*65)
