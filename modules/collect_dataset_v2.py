"""Dedicated Real-Data Collection Tool for Dataset V2 (17-Class Vocabulary).

Pipeline:
  Camera Feed -> MediaPipe Hand Landmarker -> Hand Modality Filter ->
  126-Feature Vector (Detection Order) -> Stability Hold Window ->
  Near-Duplicate / Similarity Check -> Auto Cooldown -> Append to dataset/Forproject_v2.csv

Specifications:
  - 17 Supported Classes with Standardized Modalities (11 ONE_HAND, 6 TWO_HANDS)
  - Target: 200 samples for HELLO, GOODBYE, YES; 180 samples for other 14 classes
  - Strict Modality Enforcement (Rejects invalid hand counts)
  - Hands-Free Auto Mode (600ms steady hold + 1.0s cooldown between distinct poses)
  - Near-Duplicate Frame Prevention (Euclidean feature distance threshold)
  - Manual Mode via SPACE key
  - Dry-Run / Test Mode for non-GUI automated verification

Usage:
  python modules/collect_dataset_v2.py
  python modules/collect_dataset_v2.py --sign SLEEP
  python modules/collect_dataset_v2.py --dry-run
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure root directory is on Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd

from modules.hand_detection import initialize_hands, process_hand_frame_with_handedness
from modules.landmark_features import (
    SINGLE_HAND_FEATURES,
    TWO_HAND_FEATURES,
    extract_two_hand_features,
)
from modules.validate_final_dataset import (
    EXPECTED_COLUMNS,
    EXPECTED_FEATURES,
    EXPECTED_TOTAL_COLUMNS,
    STANDARDIZED_HAND_MODALITY,
)

# ---------------------------------------------------------------------------
# Configuration & Targets
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_CSV = ROOT_DIR / "dataset" / "sumit_subject_sample.csv"
DATASET_V2_PATH = DEFAULT_OUTPUT_CSV
WINDOW_TITLE = "Sign Language AI - Dataset V2 Collector"

SIGNS_ORDER: List[str] = ["I", "YOU", "HE", "SHE", "WE", "THEY"]

SUBJECT_HAND_MODALITY: Dict[str, str] = {
    "I": "ONE_HAND",
    "YOU": "ONE_HAND",
    "HE": "ONE_HAND",
    "SHE": "ONE_HAND",
    "WE": "TWO_HANDS",
    "THEY": "ONE_HAND",
}

TARGET_SAMPLES: Dict[str, int] = {
    "I": 100, "YOU": 100, "HE": 100, "SHE": 100, "WE": 100, "THEY": 100
}

# Duplicate prevention & timing thresholds
MIN_FEATURE_DISTANCE_THRESHOLD = 0.045
AUTO_STABILITY_HOLD_SEC = 0.60  # Require holding pose steady for 600ms before auto-saving
AUTO_COOLDOWN_SEC = 1.00        # 1.0s cooldown after each capture to allow repositioning


def get_required_hand_modality(sign: str) -> str:
    """Return the required hand count for a subject sign or standard sign."""
    return SUBJECT_HAND_MODALITY.get(
        sign, STANDARDIZED_HAND_MODALITY.get(sign, "ONE_HAND")
    )


# ---------------------------------------------------------------------------
# Dataset Helper Functions
# ---------------------------------------------------------------------------

def ensure_dataset_v2() -> None:
    """Ensure the dataset exists with the expected header, migrating headerless rows."""
    DATASET_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET_V2_PATH.exists() or DATASET_V2_PATH.stat().st_size == 0:
        df_empty = pd.DataFrame(columns=EXPECTED_COLUMNS)
        df_empty.to_csv(DATASET_V2_PATH, index=False)
        return

    existing_columns = list(pd.read_csv(DATASET_V2_PATH, nrows=0).columns)
    if existing_columns == EXPECTED_COLUMNS:
        return

    headerless_df = pd.read_csv(DATASET_V2_PATH, header=None)
    if headerless_df.shape[1] != EXPECTED_TOTAL_COLUMNS:
        raise ValueError(
            f"Invalid dataset schema in {DATASET_V2_PATH}: expected "
            f"{EXPECTED_TOTAL_COLUMNS} columns, found {headerless_df.shape[1]}."
        )

    headerless_df.columns = EXPECTED_COLUMNS
    headerless_df.to_csv(DATASET_V2_PATH, index=False)


def get_current_sample_counts() -> Dict[str, int]:
    """Return dictionary of current sample count for every sign in Forproject_v2.csv."""
    ensure_dataset_v2()
    counts = {s: 0 for s in SIGNS_ORDER}
    if not DATASET_V2_PATH.exists() or DATASET_V2_PATH.stat().st_size == 0:
        return counts

    try:
        df = pd.read_csv(DATASET_V2_PATH)
        if not df.empty and "label" in df.columns:
            val_counts = df["label"].astype(str).value_counts().to_dict()
            for s in SIGNS_ORDER:
                counts[s] = int(val_counts.get(s, 0))
    except Exception as exc:
        print(f"Warning: Could not read sample counts: {exc}")
    return counts


def append_sample_v2(features: List[float], label: str) -> bool:
    """Safely append a verified 126-feature vector to Forproject_v2.csv."""
    if len(features) != EXPECTED_FEATURES:
        return False

    ensure_dataset_v2()
    row_data = list(features) + [label]
    df_row = pd.DataFrame([row_data], columns=EXPECTED_COLUMNS)
    df_row.to_csv(DATASET_V2_PATH, mode="a", header=False, index=False)
    return True


def delete_last_sample_v2(label: Optional[str] = None) -> bool:
    """Remove the most recently added sample for the specified label from Forproject_v2.csv."""
    ensure_dataset_v2()
    try:
        df = pd.read_csv(DATASET_V2_PATH)
        if df.empty:
            return False

        if label:
            matching_indices = df[df["label"] == label].index
            if len(matching_indices) == 0:
                return False
            df = df.drop(matching_indices[-1])
        else:
            df = df.iloc[:-1]

        df.to_csv(DATASET_V2_PATH, index=False)
        return True
    except Exception as exc:
        print(f"Error deleting last sample: {exc}")
        return False


def is_near_duplicate(new_features: List[float], last_features: Optional[List[float]]) -> bool:
    """Check if the new feature vector is nearly identical to the previous vector."""
    if last_features is None:
        return False
    diff = np.array(new_features, dtype=float) - np.array(last_features, dtype=float)
    dist = float(np.linalg.norm(diff))
    return dist < MIN_FEATURE_DISTANCE_THRESHOLD


# ---------------------------------------------------------------------------
# Visual Overlay & HUD
# ---------------------------------------------------------------------------

def draw_hud(
    frame: np.ndarray,
    current_sign: str,
    counts: Dict[str, int],
    new_collected_count: int,
    detected_hand_count: int,
    is_auto_mode: bool,
    status_msg: str,
    status_color: Tuple[int, int, int] = (100, 255, 100),
    hold_progress: float = 0.0,
) -> np.ndarray:
    """Render comprehensive collection HUD and feedback on camera frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top-left panel (x: 10..460, y: 10..350)
    cv2.rectangle(overlay, (10, 10), (460, 350), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, overlay)

    font = cv2.FONT_HERSHEY_SIMPLEX
    target = TARGET_SAMPLES.get(current_sign, 180)
    current_total = counts.get(current_sign, 0)
    required_mode = get_required_hand_modality(current_sign)

    y = 35
    # Header & Mode Badge
    cv2.putText(overlay, "DATASET V2 COLLECTOR", (20, y), font, 0.65, (0, 215, 255), 2)
    mode_badge = "[ AUTO ON ]" if is_auto_mode else "[ MANUAL ]"
    mode_badge_color = (0, 255, 120) if is_auto_mode else (180, 180, 180)
    cv2.putText(overlay, mode_badge, (310, y), font, 0.58, mode_badge_color, 2)

    # Sign & Modality
    y += 32
    cv2.putText(overlay, f"Sign: {current_sign}", (20, y), font, 0.62, (255, 255, 255), 2)
    y += 26
    mode_color = (120, 255, 120) if required_mode == "ONE_HAND" else (255, 200, 100)
    cv2.putText(overlay, f"Required: {required_mode}", (20, y), font, 0.50, mode_color, 1)

    # Progress Bar & Counts
    y += 28
    pct = min(1.0, current_total / max(1, target))
    cv2.putText(overlay, f"Progress: {current_total} / {target} ({int(pct*100)}%)", (20, y), font, 0.52, (220, 220, 220), 1)
    y += 10
    # Draw progress bar
    bar_w = 420
    cv2.rectangle(overlay, (20, y), (20 + bar_w, y + 8), (50, 50, 50), -1)
    fill_w = int(bar_w * pct)
    bar_color = (0, 200, 0) if pct >= 1.0 else (0, 180, 255)
    cv2.rectangle(overlay, (20, y), (20 + fill_w, y + 8), bar_color, -1)

    # Session New Count
    y += 26
    cv2.putText(overlay, f"Session New: +{new_collected_count} samples", (20, y), font, 0.50, (180, 255, 180), 1)

    # Live Hand Detection Status
    y += 26
    hand_match = (
        (required_mode == "ONE_HAND" and detected_hand_count == 1) or
        (required_mode == "TWO_HANDS" and detected_hand_count == 2)
    )
    hand_str = f"Hands in View: {detected_hand_count}"
    hand_color = (0, 255, 0) if hand_match else (0, 80, 255)
    cv2.putText(overlay, hand_str, (20, y), font, 0.50, hand_color, 2 if hand_match else 1)

    # Hold Stability Meter (for Hands-Free Auto Mode)
    y += 26
    if is_auto_mode:
        hold_bar_w = 420
        cv2.rectangle(overlay, (20, y), (20 + hold_bar_w, y + 8), (40, 40, 50), -1)
        hold_fill = int(hold_bar_w * max(0.0, min(1.0, hold_progress)))
        cv2.rectangle(overlay, (20, y), (20 + hold_fill, y + 8), (0, 255, 255), -1)
        y += 22

    # Capture Status / Message
    cv2.putText(overlay, f"Status: {status_msg}", (20, y), font, 0.48, status_color, 1)

    # Bottom Instructions Bar
    cv2.rectangle(overlay, (0, h - 35), (w, h), (10, 10, 15), -1)
    controls_txt = "[C] Auto Mode  |  [SPACE] Manual  |  [N/P] Next/Prev  |  [D] Undo  |  [Q] Exit"
    cv2.putText(overlay, controls_txt, (15, h - 12), font, 0.42, (180, 180, 180), 1)

    return overlay


# ---------------------------------------------------------------------------
# Main Interactive Collection Loop
# ---------------------------------------------------------------------------

def run_collection(initial_sign: Optional[str] = None, camera_id: int = 0) -> int:
    """Launch the interactive webcam collection pipeline."""
    ensure_dataset_v2()

    sign_index = 0
    if initial_sign and initial_sign.upper() in SIGNS_ORDER:
        sign_index = SIGNS_ORDER.index(initial_sign.upper())

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot access camera {camera_id}.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        landmarker = initialize_hands()
    except Exception as exc:
        print(f"ERROR: Failed to initialize MediaPipe: {exc}")
        cap.release()
        return 1

    print("\n" + "=" * 65)
    print("  [CAMERA] DATASET V2 INTERACTIVE COLLECTOR (WITH HANDS-FREE AUTO MODE)")
    print(f"  Saving to: {DATASET_V2_PATH}")
    print("=" * 65)
    print("  Controls:")
    print("    [C]       : Toggle Hands-Free Auto Mode (ON by default for 2-hand signs)")
    print("    [SPACE]   : Manual capture of 1 genuine sample")
    print("    [N] / [P] : Switch to Next / Previous sign")
    print("    [D]       : Undo / Delete last captured sample")
    print("    [Q] / ESC : Save and Exit")
    print("=" * 65 + "\n")

    counts = get_current_sample_counts()
    new_counts = {s: 0 for s in SIGNS_ORDER}

    current_sign = SIGNS_ORDER[sign_index]
    required_mode = get_required_hand_modality(current_sign)

    # Start with Auto Mode enabled by default for two-hand signs
    is_auto_mode = (required_mode == "TWO_HANDS")
    last_capture_time = 0.0
    hold_start_time = 0.0
    last_saved_features: Optional[List[float]] = None
    status_msg = "Ready to record"
    status_color = (200, 200, 200)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read camera frame.")
                break

            # Mirror view for natural interaction
            frame = cv2.flip(frame, 1)

            # Detect hands and draw skeleton
            ann_frame, detected_hands, _ = process_hand_frame_with_handedness(frame, landmarker)
            hand_count = len(detected_hands)

            current_sign = SIGNS_ORDER[sign_index]
            required_mode = get_required_hand_modality(current_sign)

            # Extract 126-feature vector
            features = extract_two_hand_features(detected_hands) if hand_count > 0 else None

            # Check if condition is valid for capture
            valid_modality = (
                (required_mode == "ONE_HAND" and hand_count == 1) or
                (required_mode == "TWO_HANDS" and hand_count == 2)
            )

            now = time.time()
            hold_progress = 0.0

            # ── Hands-Free Automatic Capture Engine ────────────────────────
            if is_auto_mode:
                if not valid_modality or features is None:
                    hold_start_time = 0.0
                    target_txt = "2 hands" if required_mode == "TWO_HANDS" else "1 hand"
                    status_msg = f"Waiting for {target_txt} ({hand_count} in view)"
                    status_color = (0, 80, 255)
                else:
                    time_since_last = now - last_capture_time

                    if time_since_last < AUTO_COOLDOWN_SEC:
                        hold_start_time = 0.0
                        rem = AUTO_COOLDOWN_SEC - time_since_last
                        status_msg = f"WAIT Cooldown ({rem:.1f}s) - Change pose/angle slightly"
                        status_color = (0, 180, 255)
                    elif is_near_duplicate(features, last_saved_features):
                        hold_start_time = 0.0
                        status_msg = "WARN Similar pose - Move/tilt hands slightly for variety"
                        status_color = (0, 165, 255)
                    else:
                        # Poses are valid, cooldown elapsed, and new pose is distinct!
                        if hold_start_time == 0.0:
                            hold_start_time = now

                        held_duration = now - hold_start_time
                        hold_progress = held_duration / AUTO_STABILITY_HOLD_SEC

                        if held_duration < AUTO_STABILITY_HOLD_SEC:
                            pct_int = int(hold_progress * 100)
                            status_msg = f"Hold stable... [{pct_int}%]"
                            status_color = (0, 255, 255)
                        else:
                            # 600ms stable hold verified -> Capture sample!
                            try:
                                saved = append_sample_v2(features, current_sign)
                            except PermissionError:
                                hold_start_time = 0.0
                                status_msg = "CSV locked: close Excel, then retry"
                                status_color = (0, 0, 255)
                            else:
                                if saved:
                                    counts[current_sign] = counts.get(current_sign, 0) + 1
                                    new_counts[current_sign] = new_counts.get(current_sign, 0) + 1
                                    last_saved_features = features
                                    last_capture_time = now
                                    hold_start_time = 0.0
                                    hold_progress = 0.0
                                    status_msg = f"OK CAPTURED #{counts[current_sign]}! Now change pose"
                                    status_color = (0, 255, 0)

            # Draw HUD
            hud_frame = draw_hud(
                ann_frame,
                current_sign,
                counts,
                new_counts[current_sign],
                hand_count,
                is_auto_mode,
                status_msg,
                status_color,
                hold_progress,
            )

            cv2.imshow(WINDOW_TITLE, hud_frame)
            key = cv2.waitKey(1) & 0xFF

            # Key Handling
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("n"):  # Next sign
                sign_index = (sign_index + 1) % len(SIGNS_ORDER)
                last_saved_features = None
                hold_start_time = 0.0
                req = get_required_hand_modality(SIGNS_ORDER[sign_index])
                if req == "TWO_HANDS":
                    is_auto_mode = True
                status_msg = f"Switched to {SIGNS_ORDER[sign_index]}"
                status_color = (255, 255, 255)
            elif key == ord("p"):  # Previous sign
                sign_index = (sign_index - 1) % len(SIGNS_ORDER)
                last_saved_features = None
                hold_start_time = 0.0
                req = get_required_hand_modality(SIGNS_ORDER[sign_index])
                if req == "TWO_HANDS":
                    is_auto_mode = True
                status_msg = f"Switched to {SIGNS_ORDER[sign_index]}"
                status_color = (255, 255, 255)
            elif key == ord("c"):  # Toggle Auto Mode
                is_auto_mode = not is_auto_mode
                hold_start_time = 0.0
                status_msg = "Auto Mode: ENABLED (Hands-Free)" if is_auto_mode else "Auto Mode: OFF (Manual SPACE)"
                status_color = (0, 255, 120) if is_auto_mode else (200, 200, 200)
            elif key == ord("d"):  # Undo last sample
                if delete_last_sample_v2(current_sign):
                    counts[current_sign] = max(0, counts.get(current_sign, 0) - 1)
                    new_counts[current_sign] = max(0, new_counts.get(current_sign, 0) - 1)
                    last_saved_features = None
                    hold_start_time = 0.0
                    status_msg = f"Deleted last {current_sign} sample"
                    status_color = (0, 100, 255)
            elif key == 32:  # Spacebar (Manual single capture)
                if valid_modality and features is not None:
                    if not is_near_duplicate(features, last_saved_features):
                        try:
                            saved = append_sample_v2(features, current_sign)
                        except PermissionError:
                            status_msg = "CSV locked: close Excel, then retry"
                            status_color = (0, 0, 255)
                        else:
                            if saved:
                                counts[current_sign] = counts.get(current_sign, 0) + 1
                                new_counts[current_sign] = new_counts.get(current_sign, 0) + 1
                                last_saved_features = features
                                last_capture_time = now
                                hold_start_time = 0.0
                                status_msg = f"OK Saved sample #{counts[current_sign]}"
                                status_color = (0, 255, 0)
                    else:
                        status_msg = "WARN Near-duplicate pose. Please vary hand angle or position!"
                        status_color = (0, 180, 255)
                else:
                    status_msg = f"ERROR Rejected: Requires {required_mode} ({hand_count} hands in view)"
                    status_color = (0, 0, 255)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\nSession Summary:")
    for s in SIGNS_ORDER:
        if new_counts[s] > 0:
            print(f"  • {s:<14}: +{new_counts[s]} new samples (Total: {counts[s]})")
    print(f"\nAll samples saved cleanly to: {DATASET_V2_PATH}\n")
    return 0


# ---------------------------------------------------------------------------
# CLI Entrypoint & Dry-Run
# ---------------------------------------------------------------------------

def run_dry_run_test() -> bool:
    """Perform headless dry-run verification of feature extraction and dataset routines."""
    print("Running headless dry-run verification...")
    ensure_dataset_v2()

    # 1. Test sample counts
    counts = get_current_sample_counts()
    print(f"OK Current sample counts loaded ({len(counts)} signs).")

    # 2. Test near-duplicate check
    vec1 = [0.1] * 126
    vec2 = [0.1] * 126
    vec3 = [0.5] * 126
    assert is_near_duplicate(vec1, vec2) is True, "Exact match must be flagged as duplicate"
    assert is_near_duplicate(vec1, vec3) is False, "Different poses must not be duplicate"
    print("OK Duplicate prevention logic verified.")

    # 3. Test modality rules
    assert STANDARDIZED_HAND_MODALITY["HELLO"] == "ONE_HAND"
    assert STANDARDIZED_HAND_MODALITY["PLEASE"] == "TWO_HANDS"
    assert STANDARDIZED_HAND_MODALITY["REST"] == "TWO_HANDS"
    assert get_required_hand_modality("WE") == "TWO_HANDS"
    assert get_required_hand_modality("THEY") == "ONE_HAND"
    print("OK Modality specifications verified.")

    print("\nDry-run test PASSED successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Dataset V2 Collection Tool (17-Class Vocabulary)")
    parser.add_argument("--sign", choices=SIGNS_ORDER, help="Initial sign label to collect")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="CSV path to save sample rows to")
    parser.add_argument("--dry-run", action="store_true", help="Run automated dry-run test without opening webcam")
    args = parser.parse_args()

    global DATASET_V2_PATH
    DATASET_V2_PATH = args.output.resolve() if args.output.is_absolute() else (ROOT_DIR / args.output).resolve()

    if args.dry_run:
        return 0 if run_dry_run_test() else 1

    return run_collection(initial_sign=args.sign, camera_id=args.camera)


if __name__ == "__main__":
    sys.exit(main())
