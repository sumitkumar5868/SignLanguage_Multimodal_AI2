"""Webcam + MediaPipe hand landmark collection for Phase 5.

Pipeline:
Laptop Camera -> OpenCV -> MediaPipe Hand Landmarker -> 21 Hand Landmarks ->
63 Features -> Sign Label -> CSV Dataset
"""

import time

import cv2

from modules.dataset_utils import (
    INITIAL_SIGNS,
    TARGET_SAMPLES_PER_SIGN,
    append_sample,
    count_label_samples,
    delete_last_sample,
    ensure_dataset_file,
)
from modules.hand_detection import initialize_hands, process_hand_frame
from modules.landmark_features import extract_landmark_features

WINDOW_TITLE = "SignLanguage Multimodal AI - Camera"


def get_selected_sign(key):
    """Return the selected class based on key presses 1-5."""
    mapping = {
        ord("1"): "HELLO",
        ord("2"): "I_LOVE_YOU",
        ord("3"): "I_HATE_YOU",
        ord("4"): "I_EAT",
        ord("5"): "THANK_YOU",
    }
    return mapping.get(key)


def draw_overlay(frame, current_sign, sample_count, num_hands):
    """Draw dataset collection information on the camera image with proper spacing."""
    overlay = frame.copy()
    
    # Create a semi-transparent background rectangle for readability
    # Panel dimensions: x=10 to 400, y=10 to 310
    cv2.rectangle(overlay, (10, 10), (400, 310), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, overlay)
    
    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.65
    font_thickness = 2
    color_title = (255, 255, 100)  # Yellow
    color_info = (100, 255, 255)   # Cyan
    color_status = (100, 255, 100)  # Green for detected, Red for not detected
    
    # Y positions for each line (proper spacing: ~35 pixels apart)
    y = 35
    line_height = 35
    
    # Title
    cv2.putText(overlay, "PHASE 5: DATASET COLLECTION", (20, y), font, font_size, color_title, font_thickness)
    
    # Current sign
    y += line_height
    cv2.putText(overlay, f"Current Sign: {current_sign}", (20, y), font, font_size, color_info, font_thickness)
    
    # Sample count
    y += line_height
    cv2.putText(overlay, f"Samples: {sample_count} / {TARGET_SAMPLES_PER_SIGN}", (20, y), font, font_size, color_info, font_thickness)
    
    # Hand detection status
    y += line_height
    if num_hands == 0:
        hand_status = "Hand: NOT DETECTED"
        color_hand = (0, 0, 255)  # Red
    elif num_hands == 1:
        hand_status = "Hand: DETECTED"
        color_hand = (0, 255, 0)  # Green
    else:
        hand_status = f"Hand: {num_hands} (SHOW ONE ONLY)"
        color_hand = (0, 165, 255)  # Orange
    
    cv2.putText(overlay, hand_status, (20, y), font, font_size, color_hand, font_thickness)
    
    # Landmarks count
    y += line_height
    if num_hands == 1:
        cv2.putText(overlay, "Landmarks: 21", (20, y), font, font_size, color_status, font_thickness)
    else:
        cv2.putText(overlay, "Landmarks: 0", (20, y), font, font_size, (100, 100, 100), font_thickness)
    
    # Features count
    y += line_height
    if num_hands == 1:
        cv2.putText(overlay, "Features: 63", (20, y), font, font_size, color_status, font_thickness)
    else:
        cv2.putText(overlay, "Features: 0", (20, y), font, font_size, (100, 100, 100), font_thickness)
    
    # Control instructions
    y += line_height + 10
    cv2.putText(overlay, "1-5: Select Sign", (20, y), font, 0.6, (200, 200, 200), 1)
    
    y += 25
    cv2.putText(overlay, "SPACE: Capture Sample", (20, y), font, 0.6, (200, 200, 200), 1)
    
    y += 25
    cv2.putText(overlay, "BACKSPACE: Delete Last Sample", (20, y), font, 0.6, (200, 200, 200), 1)
    
    y += 25
    cv2.putText(overlay, "Q: Quit", (20, y), font, 0.6, (200, 200, 200), 1)
    
    return overlay


def run_camera():
    """Open the webcam and collect a labeled landmark dataset using key presses."""
    print("========================================")
    print(" SIGN LANGUAGE MULTIMODAL AI")
    print("========================================")
    print()
    print("Phase 1: Project Initialization")
    print("Status: SUCCESS")
    print()
    print("Phase 2: Webcam + OpenCV")
    print("Status: SUCCESS")
    print()
    print("Phase 3: Hand Detection")
    print("Status: SUCCESS")
    print()
    print("Phase 4: Landmark Feature Extraction")
    print("Status: SUCCESS")
    print()
    print("Phase 5: Dataset Collection")
    print("Status: STARTING")
    print()
    print("Dataset:")
    print("dataset/sign_landmarks.csv")
    print()
    print("Signs:")
    print("1. HELLO")
    print("2. I_LOVE_YOU")
    print("3. I_HATE_YOU")
    print("4. I_EAT")
    print("5. THANK_YOU")
    print()
    print("Target: 500 samples per sign")
    print()
    print("Controls:")
    print("1-5 → Select Sign")
    print("SPACE → Capture Sample")
    print("BACKSPACE → Delete Last Sample")
    print("Q → Quit")
    print()
    print("Starting camera...")

    ensure_dataset_file()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return 1

    try:
        hands = initialize_hands()
    except RuntimeError as exc:
        print(str(exc))
        camera.release()
        return 1

    current_sign = "HELLO"
    last_space_press = 0.0
    last_status_message = ""
    sample_count = 0
    last_count_update = 0.0
    frame_count = 0

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1000, 700)
    cv2.moveWindow(WINDOW_TITLE, 50, 50)

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("ERROR: Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            processed_frame, detected_hands = process_hand_frame(frame, hands)

            # Update sample count only every second (not every frame) to avoid I/O blocking
            now = time.time()
            if now - last_count_update >= 1.0:
                sample_count = count_label_samples(current_sign)
                last_count_update = now

            overlay = draw_overlay(processed_frame, current_sign, sample_count, len(detected_hands))
            cv2.imshow(WINDOW_TITLE, overlay)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                selected = get_selected_sign(key)
                if selected is not None:
                    current_sign = selected
                    sample_count = count_label_samples(current_sign)
                    last_count_update = now
                    last_status_message = f"Current Sign: {current_sign}"
                    print(last_status_message)
                    print(f"Samples: {sample_count} / {TARGET_SAMPLES_PER_SIGN}")

            if key == 32:
                now = time.time()
                if now - last_space_press < 0.4:
                    continue
                last_space_press = now

                if len(detected_hands) != 1:
                    if len(detected_hands) == 0:
                        print("No Hand Detected")
                        print("Sample Not Saved")
                    elif len(detected_hands) > 1:
                        print("Please show one hand only.")
                        print("Sample Not Saved")
                    last_status_message = "Sample Not Saved"
                    continue

                features = extract_landmark_features(detected_hands[0])
                if features is None:
                    print("Invalid Landmark Data")
                    print("Sample Not Saved")
                    last_status_message = "Invalid Landmark Data"
                    continue

                if len(features) != 63:
                    print("Invalid Feature Count")
                    print("Expected: 63")
                    print("Sample Not Saved")
                    last_status_message = "Invalid Feature Count"
                    continue

                try:
                    append_sample(current_sign, features)
                    sample_count = count_label_samples(current_sign)
                    last_count_update = now
                    print("Sample Saved")
                    print(f"Label: {current_sign}")
                    print(f"Samples: {sample_count} / {TARGET_SAMPLES_PER_SIGN}")
                    last_status_message = "Sample Saved"
                except ValueError as exc:
                    print(str(exc))
                    print("Sample Not Saved")
                    last_status_message = "Sample Not Saved"

            if key == 8:  # BACKSPACE key
                success, message = delete_last_sample(current_sign)
                print(message)
                if success:
                    sample_count = count_label_samples(current_sign)
                    last_count_update = now
                    print(f"Samples: {sample_count} / {TARGET_SAMPLES_PER_SIGN}")
                    last_status_message = message
                else:
                    last_status_message = message

            if key == ord("q") or key == ord("Q"):
                print("Closing camera...")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        hands.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(run_camera())
