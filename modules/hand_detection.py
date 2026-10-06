"""Phase 3 and Phase 4 hand detection using the current MediaPipe Tasks API."""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    drawing_styles,
    drawing_utils,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "hand_landmarker.task"
OFFICIAL_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def initialize_hands(
    model_path=None,
    min_detection_confidence: float = 0.30,
    min_presence_confidence: float = 0.30,
):
    """Create and return the current MediaPipe Hand Landmarker."""
    target_path = Path(model_path) if model_path else MODEL_PATH

    if not target_path.exists():
        raise RuntimeError(
            "ERROR: Missing MediaPipe Hand Landmarker model file.\n"
            f"Place the official model here: {target_path}\n"
            f"Official download URL: {OFFICIAL_MODEL_URL}"
        )

    try:
        options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(target_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
        )
        return HandLandmarker.create_from_options(options)
    except Exception as exc:
        raise RuntimeError(f"ERROR: MediaPipe hand detection could not be initialized: {exc}") from exc


def detect_hands_fast(frame, hand_landmarker):
    """Ultra-fast hand detection without frame copying or landmark drawing overhead."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = hand_landmarker.detect(mp_image)

    detected_hands = []
    handedness_list = []

    if result.hand_landmarks:
        if hasattr(result, "handedness") and result.handedness:
            for hand_categories in result.handedness:
                if hand_categories and len(hand_categories) > 0:
                    cat = hand_categories[0]
                    handedness_list.append({
                        "label": getattr(cat, "category_name", "") or getattr(cat, "display_name", "Unknown"),
                        "score": float(getattr(cat, "score", 0.0)),
                    })
                else:
                    handedness_list.append({"label": "Unknown", "score": 0.0})

        detected_hands = list(result.hand_landmarks)

    return detected_hands, handedness_list


def process_hand_frame(frame, hand_landmarker):
    """Detect hands in a frame, draw landmarks, and return the detected landmark sets."""
    annotated_frame, detected_hands, _ = process_hand_frame_with_handedness(frame, hand_landmarker)
    return annotated_frame, detected_hands


def process_hand_frame_with_handedness(frame, hand_landmarker):
    """Detect up to 2 hands, draw landmarks, and return landmarks + handedness metadata."""
    annotated_frame = frame.copy()
    detected_hands = []
    handedness_list = []

    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        # Extract handedness if provided by MediaPipe
        if hasattr(result, "handedness") and result.handedness:
            for hand_categories in result.handedness:
                if hand_categories and len(hand_categories) > 0:
                    cat = hand_categories[0]
                    handedness_list.append({
                        "label": getattr(cat, "category_name", "") or getattr(cat, "display_name", "Unknown"),
                        "score": float(getattr(cat, "score", 0.0)),
                    })
                else:
                    handedness_list.append({"label": "Unknown", "score": 0.0})

        for hand_landmarks in result.hand_landmarks:
            detected_hands.append(hand_landmarks)
            drawing_utils.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
            )

    return annotated_frame, detected_hands, handedness_list

