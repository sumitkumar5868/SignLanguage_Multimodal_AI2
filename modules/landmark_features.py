"""Feature Extraction Module for One-Hand (63) and Two-Hand (126) Sign Recognition.

Specifications:
- One Hand:  21 landmarks × 3 coordinates (x, y, z) = 63 features
- Two Hands: 42 landmarks × 3 coordinates (x, y, z) = 126 features
  - h1 slot: Indices 0..62   (21 × 3 = 63 floats)
  - h2 slot: Indices 63..125 (21 × 3 = 63 floats)

Hand Ordering Convention — DETECTION ORDER (matches dataset/Forproject.csv):
- When 2 hands are detected:
  - First  detected hand → h1 slot (indices 0..62)
  - Second detected hand → h2 slot (indices 63..125)
  - Handedness (left/right) is NOT used for slot assignment.
- When 1 hand is detected:
  - The single hand → h1 slot (indices 0..62)
  - h2 slot → 63 zeros
  - Handedness (left/right) is NOT used for slot assignment.
- When 0 hands are detected:
  - Returns None.

Why detection order?
  dataset/Forproject.csv was collected in detection order (h1 = first detected
  hand, h2 = second detected hand or zeros).  Using the same convention at
  inference time ensures the 126-feature vectors seen during training exactly
  match the 126-feature vectors produced at runtime.

Wrist-Relative Features (position-independent):
  extract_single_hand_wristrel()  → 63 floats, each landmark relative to wrist
  extract_two_hand_wristrel()     → 126 floats, same detection-order convention
  These are used by the wristrel model pipeline (sign_language_model_wristrel_17class.pkl)
  which is position-independent — the hand can be anywhere on screen.
"""

from typing import Any, Dict, List, Optional, Tuple

SINGLE_HAND_FEATURES = 63
TWO_HAND_FEATURES = 126
NUM_LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3


def extract_single_hand_features(hand_landmarks) -> Optional[List[float]]:
    """Extract a 63-value feature vector from a single hand's landmarks.

    Args:
        hand_landmarks: Iterable of 21 landmark objects with .x, .y, .z attributes.

    Returns:
        List of 63 floats, or None if extraction fails.
    """
    if hand_landmarks is None:
        return None

    try:
        features = []
        for landmark in hand_landmarks:
            features.extend([float(landmark.x), float(landmark.y), float(landmark.z)])

        if len(features) != SINGLE_HAND_FEATURES:
            return None

        return features
    except (AttributeError, TypeError, ValueError):
        return None


# Backward-compatible alias for existing 1-hand pipeline
def extract_landmark_features(hand_landmarks) -> Optional[List[float]]:
    """Flatten MediaPipe hand landmarks into a 63-value feature vector (1-hand legacy API)."""
    return extract_single_hand_features(hand_landmarks)


def extract_two_hand_features(
    detected_hands: List[Any],
    handedness_list: Optional[List[Dict[str, Any]]] = None,  # noqa: ARG001 — kept for API compat, not used for ordering
) -> Optional[List[float]]:
    """Extract a 126-feature vector using DETECTION-ORDER hand assignment.

    Layout (matches dataset/Forproject.csv exactly):
      [0..62]:   h1 — first  detected hand (21 landmarks × 3 [x, y, z])
      [63..125]: h2 — second detected hand (21 landmarks × 3 [x, y, z]),
                      or 63 zeros when only one hand is present.

    Convention — DETECTION ORDER (not left/right):
      Hand slot assignment is determined by MediaPipe detection order, NOT
      by handedness (left/right) classification.  This matches the ordering
      used when Forproject.csv was collected, ensuring training and inference
      produce identical 126-feature vectors.

      h1 = detected_hands[0]  (always the first hand MediaPipe returns)
      h2 = detected_hands[1]  (second hand if present, else 63 zeros)

    Args:
        detected_hands: List of hand landmark sets (length 0, 1, or 2).
        handedness_list: Accepted for API backward-compatibility but NOT used
                         for slot ordering.  Slot assignment is detection-order
                         only so that runtime vectors match Forproject.csv.

    Returns:
        List of 126 floats, or None if no hands detected.
    """
    if not detected_hands:
        return None

    hand_count = len(detected_hands)
    if hand_count == 0:
        return None

    # h1 slot: first detected hand | h2 slot: second detected hand or zeros
    h1_features = [0.0] * SINGLE_HAND_FEATURES
    h2_features = [0.0] * SINGLE_HAND_FEATURES

    # ── Case 1: Two (or more) hands detected ────────────────────────────────
    # Detection order: detected_hands[0] → h1, detected_hands[1] → h2.
    # Left/right handedness is deliberately ignored for slot assignment.
    if hand_count >= 2:
        h1_feats = extract_single_hand_features(detected_hands[0])
        h2_feats = extract_single_hand_features(detected_hands[1])

        if not h1_feats or not h2_feats:
            return None

        # Assign in detection order — matches Forproject.csv convention
        h1_features = h1_feats
        h2_features = h2_feats

        return h1_features + h2_features

    # ── Case 2: Exactly one hand detected ───────────────────────────────────
    # The single hand always goes into h1; h2 is zero-padded.
    # Left/right handedness is deliberately ignored — matches Forproject.csv
    # where h1 is always filled and h2 is always zeros for single-hand rows.
    if hand_count == 1:
        h1_feats = extract_single_hand_features(detected_hands[0])
        if not h1_feats:
            return None

        # h1 = detected hand, h2 = zeros (detection-order convention)
        h1_features = h1_feats
        # h2_features already initialised to [0.0] × 63

        return h1_features + h2_features

    return None


def extract_features_by_expected_dim(
    detected_hands: List[Any],
    handedness_list: Optional[List[Dict[str, Any]]] = None,
    expected_dim: int = 63,
) -> Optional[List[float]]:
    """Adaptive feature extractor matching the loaded model's expected input dimension.

    Args:
        detected_hands: List of detected hand landmark sets.
        handedness_list: Handedness classifications.
        expected_dim: 63 (1-hand model) or 126 (2-hand model).

    Returns:
        Feature list matching expected_dim, or None if incompatible/missing.
    """
    if not detected_hands:
        return None

    if expected_dim == SINGLE_HAND_FEATURES:
        # Legacy/Current 1-hand model: requires at least 1 hand, uses hand 0
        return extract_single_hand_features(detected_hands[0])

    elif expected_dim == TWO_HAND_FEATURES:
        # 2-hand model: uses 126-feature detection-order vector
        # (h1=first detected hand, h2=second detected hand or zeros)
        return extract_two_hand_features(detected_hands, handedness_list)

    else:
        # Fallback: attempt 63 if only 1 hand or 126 if 2 hands
        if len(detected_hands) == 1:
            return extract_single_hand_features(detected_hands[0])
        return extract_two_hand_features(detected_hands, handedness_list)


# ===========================================================================
# WRIST-RELATIVE FEATURE EXTRACTION (position-independent pipeline)
# ===========================================================================

def _make_wristrel_63(hand_landmarks) -> Optional[List[float]]:
    """Convert 21 MediaPipe landmarks to 63 wrist-relative floats.

    Each landmark's (x, y, z) is subtracted from the wrist landmark (index 0)
    so the result is independent of where on screen the hand is held.

    Args:
        hand_landmarks: Iterable of 21 landmark objects with .x, .y, .z
                        OR a flat list of 63 floats [x0,y0,z0, x1,y1,z1, ...].

    Returns:
        List of 63 floats (wrist-relative), or None on failure.
    """
    if hand_landmarks is None:
        return None

    try:
        # Accept both landmark objects and flat float lists
        first = hand_landmarks[0]
        if hasattr(first, 'x'):
            raw = []
            for lm in hand_landmarks:
                raw.extend([float(lm.x), float(lm.y), float(lm.z)])
        else:
            raw = [float(v) for v in hand_landmarks]

        if len(raw) != SINGLE_HAND_FEATURES:
            return None

        # Wrist is landmark 0 → raw[0], raw[1], raw[2]
        wx, wy, wz = raw[0], raw[1], raw[2]

        # Landmark 9 is Middle MCP (knuckle) → scale invariant distance
        mx = raw[27] - wx
        my = raw[28] - wy
        mz = raw[29] - wz
        scale = (mx * mx + my * my + mz * mz) ** 0.5
        if scale < 1e-4:
            scale = 1.0

        relative = []
        for i in range(NUM_LANDMARKS_PER_HAND):
            base = i * COORDINATES_PER_LANDMARK
            relative.append((raw[base]     - wx) / scale)  # scale-normalized x
            relative.append((raw[base + 1] - wy) / scale)  # scale-normalized y
            relative.append((raw[base + 2] - wz) / scale)  # scale-normalized z

        return relative
    except (AttributeError, TypeError, ValueError, IndexError):
        return None


def extract_single_hand_wristrel(hand_landmarks) -> Optional[List[float]]:
    """Extract 63 wrist-relative features from a single hand.

    Drop-in replacement for extract_single_hand_features() — same output
    shape (63 floats) but position-independent: the hand can be held
    anywhere on screen and the features will remain the same.

    Args:
        hand_landmarks: 21 MediaPipe landmark objects with .x/.y/.z
                        OR flat list of 63 floats.

    Returns:
        List of 63 floats (wrist-relative), or None on failure.
    """
    return _make_wristrel_63(hand_landmarks)


def extract_two_hand_wristrel(
    detected_hands: List[Any],
    handedness_list: Optional[List[Any]] = None,  # noqa: ARG001 — kept for API compat
) -> Optional[List[float]]:
    """Extract 126 wrist-relative features using detection-order convention.

    Same slot ordering as extract_two_hand_features():
      [0..62]:   h1 — first  detected hand (wrist-relative)
      [63..125]: h2 — second detected hand (wrist-relative), or 63 zeros.

    Each hand's features are independently made wrist-relative (each hand
    uses its own wrist as the reference point), so both hands can be held
    anywhere on screen.

    Args:
        detected_hands: List of hand landmark sets (length 0, 1, or 2).
        handedness_list: Ignored; kept for API compatibility.

    Returns:
        List of 126 floats (wrist-relative, detection-order), or None.
    """
    if not detected_hands:
        return None

    hand_count = len(detected_hands)
    h2_zeros = [0.0] * SINGLE_HAND_FEATURES

    if hand_count >= 2:
        h1 = _make_wristrel_63(detected_hands[0])
        h2 = _make_wristrel_63(detected_hands[1])
        if h1 is None or h2 is None:
            return None
        return h1 + h2

    if hand_count == 1:
        h1 = _make_wristrel_63(detected_hands[0])
        if h1 is None:
            return None
        return h1 + h2_zeros  # h2 stays zeros

    return None
