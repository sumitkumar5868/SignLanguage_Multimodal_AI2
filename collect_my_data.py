"""
Fast Personal Data Collector — Wrist-Relative Features
=======================================================
Collect YOUR OWN gesture data. Saves to: dataset/my_fresh_dataset.csv

Controls:
  SPACE  — Capture current pose manually
  C      — Toggle Auto Mode (captures every 0.5s)
  N / P  — Next / Previous sign
  D      — Delete last captured sample
  Q/ESC  — Quit

Target: 150 samples per sign
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from modules.hand_detection import initialize_hands, process_hand_frame_with_handedness
from modules.landmark_features import extract_two_hand_wristrel
from modules.validate_final_dataset import EXPECTED_COLUMNS, STANDARDIZED_HAND_MODALITY

DATASET_OUT   = ROOT / "dataset" / "my_fresh_dataset.csv"
TARGET        = 150
AUTO_HOLD_SEC = 0.50
AUTO_COOL_SEC = 0.80
DUP_THRESH    = 0.03

SIGNS_ORDER = [
    "HELLO", "GOODBYE", "YES", "NO", "PLEASE", "THANK_YOU", "SORRY",
    "HELP", "WATER", "FOOD", "BATHROOM", "PAIN", "SICK", "SLEEP",
    "REST", "STOP", "EMERGENCY"
]

GESTURE_HINT = {
    "HELLO":     "Wave hand gently near face level",
    "GOODBYE":   "Raise & wave hand AWAY from body",
    "YES":       "Fist with thumb up, bob up-down",
    "NO":        "Shake INDEX FINGER side to side",
    "PLEASE":    "[TWO HANDS] Press palms together (prayer pose)",
    "THANK_YOU": "Touch chest with flat hand/fingertips",
    "SORRY":     "Fist on CHEST in circular motion",
    "HELP":      "[TWO HANDS] Raise open palm, other hand grips wrist",
    "WATER":     "W shape: 3 fingers extended (index+mid+ring)",
    "FOOD":      "Bring ALL fingertips together to mouth",
    "BATHROOM":  "T shape: thumb between index+middle finger",
    "PAIN":      "Point one finger to area of pain",
    "SICK":      "[TWO HANDS] Middle fingers touch forehead+stomach",
    "SLEEP":     "[TWO HANDS] Clasp hands together, rest head on them",
    "REST":      "[TWO HANDS] Make T/cross shape with both hands",
    "STOP":      "[TWO HANDS] Both open palms facing camera (Push forward STOP)",
    "EMERGENCY": "[TWO HANDS] Cross arms in X on chest",
}

def ensure_dataset():
    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET_OUT.exists():
        pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(DATASET_OUT, index=False)

def get_counts():
    ensure_dataset()
    counts = {s: 0 for s in SIGNS_ORDER}
    try:
        df = pd.read_csv(DATASET_OUT)
        if not df.empty and "label" in df.columns:
            vc = df["label"].value_counts().to_dict()
            for s in SIGNS_ORDER:
                counts[s] = int(vc.get(s, 0))
    except Exception:
        pass
    return counts

def save_sample(features, label):
    ensure_dataset()
    row = list(features) + [label]
    pd.DataFrame([row], columns=EXPECTED_COLUMNS).to_csv(
        DATASET_OUT, mode="a", header=False, index=False
    )

def delete_last(label):
    try:
        df = pd.read_csv(DATASET_OUT)
        idx = df[df["label"] == label].index
        if len(idx):
            df = df.drop(idx[-1])
            df.to_csv(DATASET_OUT, index=False)
            return True
    except Exception:
        pass
    return False

def is_dup(a, b):
    if a is None or b is None:
        return False
    return float(np.linalg.norm(np.array(a) - np.array(b))) < DUP_THRESH

def draw_hud(frame, sign, counts, hand_count, is_auto, status_msg, status_color, hold_pct):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    modality = STANDARDIZED_HAND_MODALITY.get(sign, "ONE_HAND")
    count    = counts.get(sign, 0)
    pct      = count / TARGET * 100

    cv2.putText(frame, f"SIGN: {sign}", (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 180), 2)
    bar_color = (0, 255, 80) if count >= TARGET else (0, 200, 255)
    cv2.putText(frame, f"{count}/{TARGET} ({pct:.0f}%)", (w - 230, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, bar_color, 2)

    bar_w = int((w - 30) * min(pct / 100, 1.0))
    cv2.rectangle(frame, (15, 50), (w - 15, 65), (50, 50, 50), -1)
    cv2.rectangle(frame, (15, 50), (15 + bar_w, 65), bar_color, -1)

    if is_auto and hold_pct > 0:
        hl = int((w - 30) * min(hold_pct, 1.0))
        cv2.rectangle(frame, (15, 66), (15 + hl, 73), (0, 255, 255), -1)

    mode_col = (0, 255, 120) if is_auto else (180, 180, 180)
    mode_txt = "AUTO ●" if is_auto else "MANUAL ○"
    need_txt = "2 hands" if modality == "TWO_HANDS" else "1 hand"
    cv2.putText(frame, f"Mode: {mode_txt}  |  Hands: {hand_count}  |  Need: {need_txt}",
                (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_col, 1)

    hint = GESTURE_HINT.get(sign, "")
    cv2.putText(frame, hint, (15, h - 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 100), 1)
    cv2.putText(frame, status_msg, (15, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, status_color, 2)
    cv2.putText(frame, "SPACE=Capture  C=Auto  N=Next  P=Prev  D=Undo  Q=Quit",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1)

    # Sidebar — all signs progress
    sx = w - 200
    for i, s in enumerate(SIGNS_ORDER):
        y   = 105 + i * 24
        cnt = counts.get(s, 0)
        col = (0, 255, 80) if cnt >= TARGET else (0, 180, 255) if cnt > 0 else (70, 70, 70)
        mk  = "OK" if cnt >= TARGET else f"{cnt:3d}"
        bold = 2 if s == sign else 1
        cv2.putText(frame, f"{s:<12} {mk}", (sx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, bold)
    return frame

def main():
    print("=" * 58)
    print("  FRESH PERSONAL DATASET COLLECTOR (Wrist-Relative)")
    print(f"  Target  : {TARGET} samples per sign")
    print(f"  Output  : {DATASET_OUT}")
    print("=" * 58)
    print()
    print("CONTROLS:")
    print("  SPACE  — Capture single sample")
    print("  C      — Toggle Auto-capture (0.5s hold)")
    print("  N / P  — Next / Previous sign")
    print("  D      — Delete last sample")
    print("  Q/ESC  — Quit\n")

    ensure_dataset()
    counts         = get_counts()
    sign_index     = 0
    is_auto        = False
    hold_start     = 0.0
    last_capture_t = 0.0
    last_feats     = None
    status_msg     = "Press SPACE to capture  |  C for Auto mode"
    status_color   = (200, 200, 200)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam not found.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    landmarker = initialize_hands()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Match training convention
            ann_frame, detected_hands, _ = process_hand_frame_with_handedness(frame, landmarker)
            hand_count = len(detected_hands)

            sign     = SIGNS_ORDER[sign_index]
            modality = STANDARDIZED_HAND_MODALITY.get(sign, "ONE_HAND")
            need2    = (modality == "TWO_HANDS")
            features = extract_two_hand_wristrel(detected_hands) if hand_count > 0 else None
            valid    = ((need2 and hand_count == 2) or (not need2 and hand_count == 1)) and features is not None

            now      = time.time()
            hold_pct = 0.0

            if is_auto:
                if not valid:
                    hold_start   = 0.0
                    status_msg   = f"Show {'2' if need2 else '1'} hand(s) — detected: {hand_count}"
                    status_color = (0, 80, 255)
                elif (now - last_capture_t) < AUTO_COOL_SEC:
                    hold_start   = 0.0
                    rem          = AUTO_COOL_SEC - (now - last_capture_t)
                    status_msg   = f"Cooldown {rem:.1f}s — change pose slightly"
                    status_color = (0, 180, 255)
                elif is_dup(features, last_feats):
                    hold_start   = 0.0
                    status_msg   = "Same pose! Tilt/move hand slightly"
                    status_color = (0, 165, 255)
                else:
                    if hold_start == 0.0:
                        hold_start = now
                    held     = now - hold_start
                    hold_pct = held / AUTO_HOLD_SEC
                    if held < AUTO_HOLD_SEC:
                        status_msg   = f"Hold steady... {int(hold_pct*100)}%"
                        status_color = (0, 255, 255)
                    else:
                        save_sample(features, sign)
                        counts[sign]   = counts.get(sign, 0) + 1
                        last_feats     = features
                        last_capture_t = now
                        hold_start     = 0.0
                        status_msg     = f"✓ CAPTURED #{counts[sign]}!"
                        status_color   = (0, 255, 0)

            hud = draw_hud(ann_frame, sign, counts, hand_count,
                           is_auto, status_msg, status_color, hold_pct)
            cv2.imshow("Sign Language — My Data Collector", hud)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("n"):
                sign_index   = (sign_index + 1) % len(SIGNS_ORDER)
                last_feats   = None; hold_start = 0.0
                status_msg   = f"-> {SIGNS_ORDER[sign_index]}"
                status_color = (255, 255, 255)
            elif key == ord("p"):
                sign_index   = (sign_index - 1) % len(SIGNS_ORDER)
                last_feats   = None; hold_start = 0.0
                status_msg   = f"<- {SIGNS_ORDER[sign_index]}"
                status_color = (255, 255, 255)
            elif key == ord("c"):
                is_auto      = not is_auto; hold_start = 0.0
                status_msg   = "AUTO mode ON (0.5s hold)" if is_auto else "MANUAL mode (press SPACE)"
                status_color = (0, 255, 120) if is_auto else (200, 200, 200)
            elif key == ord("d"):
                if delete_last(sign):
                    counts[sign] = max(0, counts.get(sign, 0) - 1)
                    last_feats   = None
                    status_msg   = f"Deleted last {sign} sample"
                    status_color = (0, 100, 255)
            elif key == 32:
                if valid and not is_dup(features, last_feats):
                    save_sample(features, sign)
                    counts[sign]   = counts.get(sign, 0) + 1
                    last_feats     = features; last_capture_t = now; hold_start = 0.0
                    status_msg     = f"✓ Saved #{counts[sign]}"
                    status_color   = (0, 255, 0)
                elif not valid:
                    status_msg   = f"Need {'2' if need2 else '1'} hand(s) ({hand_count} detected)"
                    status_color = (0, 0, 255)
                else:
                    status_msg   = "Same pose! Change angle slightly"
                    status_color = (0, 165, 255)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 58)
    print("SESSION COMPLETE:")
    total = 0
    for s in SIGNS_ORDER:
        cnt  = counts.get(s, 0); total += cnt
        done = "OK" if cnt >= TARGET else "--"
        print(f"  [{done}] {s:<14}: {cnt}/{TARGET}")
    print(f"\n  Total samples saved: {total}")
    print(f"  File: {DATASET_OUT}")
    print("=" * 58)
    return 0

if __name__ == "__main__":
    sys.exit(main())
