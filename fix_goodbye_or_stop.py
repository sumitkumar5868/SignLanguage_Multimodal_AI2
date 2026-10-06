"""
Single Sign Re-Collector & Auto-Retrainer
========================================
Quickly re-collect samples for a single conflicting sign (e.g. GOODBYE or STOP)
without losing any of your other 16 signs!

Usage:
  venv/bin/python3 fix_goodbye_or_stop.py --sign GOODBYE
  or
  venv/bin/python3 fix_goodbye_or_stop.py --sign STOP
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/rohitroy/Downloads/SignLanguage_Multimodal_AI")
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from modules.hand_detection import initialize_hands, process_hand_frame_with_handedness
from modules.landmark_features import extract_two_hand_wristrel
from modules.validate_final_dataset import EXPECTED_COLUMNS, STANDARDIZED_HAND_MODALITY

DATASET_PATH = ROOT / "dataset" / "my_fresh_dataset.csv"
TARGET_SAMPLES = 100
AUTO_HOLD_SEC  = 0.40
AUTO_COOL_SEC  = 0.50
DUP_THRESH     = 0.025

GESTURE_GUIDES = {
    "GOODBYE": (
        "GOODBYE: Half-curl 4 fingers forward (like flapping/waving bye-bye) OR tilt hand 45 deg sideways",
        "Keep fingers slightly bent forward at knuckles — distinct from flat stiff STOP!",
    ),
    "STOP": (
        "STOP: [TWO HANDS] Both open palms facing camera ✋✋ (Push forward)",
        "Show both hands facing camera with fingers spread open — 100% distinct from Goodbye!",
    ),
}

def is_dup(a, b):
    if a is None or b is None:
        return False
    return float(np.linalg.norm(np.array(a) - np.array(b))) < DUP_THRESH

def main():
    parser = argparse.ArgumentParser(description="Re-collect one sign to fix collision.")
    parser.add_argument("--sign", default="STOP", choices=["GOODBYE", "STOP"],
                        help="Which sign to re-record (default: STOP)")
    args = parser.parse_args()
    sign = args.sign

    guide_title, guide_sub = GESTURE_GUIDES.get(sign, ("", ""))
    modality = STANDARDIZED_HAND_MODALITY.get(sign, "TWO_HANDS" if sign == "STOP" else "ONE_HAND")
    need2 = (modality == "TWO_HANDS")

    print("=" * 65)
    print(f"  RE-RECORDING CONFLICTING SIGN: {sign} ({'2 HANDS' if need2 else '1 HAND'})")
    print(f"  Target: {TARGET_SAMPLES} fresh samples")
    print(f"  Guide:  {guide_title}")
    print("=" * 65)

    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found.")
        return 1

    df = pd.read_csv(DATASET_PATH)
    old_count = len(df[df["label"] == sign])
    print(f"Current samples for {sign}: {old_count}")

    # Remove old samples of this sign
    df_clean = df[df["label"] != sign].copy()
    df_clean.to_csv(DATASET_PATH, index=False)
    print(f"✓ Removed {old_count} old samples of {sign}. Other signs preserved ({len(df_clean)} rows remaining).")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam not accessible.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    landmarker = initialize_hands()

    collected = 0
    hold_start = 0.0
    last_cap_t = 0.0
    last_feats = None

    print(f"\nStarting camera. Need {'2 hands' if need2 else '1 hand'}.")
    print("Hold steady for AUTO capture (or press SPACE). Press Q when done.\n")

    try:
        while collected < TARGET_SAMPLES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            ann_frame, hands, _ = process_hand_frame_with_handedness(frame, landmarker)
            hand_count = len(hands)

            features = extract_two_hand_wristrel(hands) if hand_count > 0 else None
            valid = ((need2 and hand_count == 2) or (not need2 and hand_count == 1)) and (features is not None)

            now = time.time()
            hold_pct = 0.0
            status_msg = "Hold pose steady..."
            status_col = (0, 255, 255)

            if not valid:
                hold_start = 0.0
                status_msg = f"Show {'2 hands' if need2 else '1 hand'} clearly to camera"
                status_col = (0, 80, 255)
            elif (now - last_cap_t) < AUTO_COOL_SEC:
                hold_start = 0.0
                status_msg = "Change position/tilt slightly..."
                status_col = (0, 180, 255)
            elif is_dup(features, last_feats):
                hold_start = 0.0
                status_msg = "Move/tilt hand slightly..."
                status_col = (0, 165, 255)
            else:
                if hold_start == 0.0:
                    hold_start = now
                held = now - hold_start
                hold_pct = held / AUTO_HOLD_SEC
                if held < AUTO_HOLD_SEC:
                    status_msg = f"Holding... {int(hold_pct*100)}%"
                    status_col = (0, 255, 255)
                else:
                    # Capture!
                    row = list(features) + [sign]
                    pd.DataFrame([row], columns=EXPECTED_COLUMNS).to_csv(
                        DATASET_PATH, mode="a", header=False, index=False
                    )
                    collected += 1
                    last_feats = features
                    last_cap_t = now
                    hold_start = 0.0
                    status_msg = f"✓ CAPTURED #{collected}/{TARGET_SAMPLES}"
                    status_col = (0, 255, 0)

            # Draw HUD
            h, w = ann_frame.shape[:2]
            cv2.rectangle(ann_frame, (0, 0), (w, 100), (20, 20, 20), -1)
            pct = collected / TARGET_SAMPLES * 100
            bar_w = int((w - 30) * min(pct / 100, 1.0))
            cv2.putText(ann_frame, f"RE-RECORDING: {sign}", (15, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2)
            cv2.putText(ann_frame, f"{collected}/{TARGET_SAMPLES} ({pct:.0f}%)", (w - 240, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 80), 2)
            cv2.rectangle(ann_frame, (15, 52), (w - 15, 66), (50, 50, 50), -1)
            cv2.rectangle(ann_frame, (15, 52), (15 + bar_w, 66), (0, 255, 80), -1)

            cv2.putText(ann_frame, guide_title[:75], (15, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 100), 1)
            cv2.putText(ann_frame, status_msg, (15, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_col, 2)

            cv2.imshow("Re-Record Conflicting Sign", ann_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key == 32:  # SPACE manual capture
                if valid and features is not None:
                    row = list(features) + [sign]
                    pd.DataFrame([row], columns=EXPECTED_COLUMNS).to_csv(
                        DATASET_PATH, mode="a", header=False, index=False
                    )
                    collected += 1
                    last_feats = features
                    last_cap_t = now

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nCollected {collected} samples for {sign}.")
    if collected >= 30:
        print("\n" + "="*60)
        print("  AUTO-RETRAINING MODEL WITH UPDATED DATA...")
        print("="*60)
        subprocess.run([sys.executable, str(ROOT / "train_my_model.py")], check=True)
        print("\n✓ Retraining completed! Model is updated.")
    else:
        print("Less than 30 samples captured; skipping auto-retrain.")

if __name__ == "__main__":
    main()
