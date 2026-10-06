"""Sign language model inference module with 1-Hand (63) and 2-Hand (126) compatibility.

Loads the trained RandomForestClassifier, StandardScaler, and LabelEncoder
once at import time (or lazily on first call) and exposes predict() and predict_detailed()
functions for real-time sign recognition.

Architecture & Compatibility:
  - Current model (63 features):
    - 1 hand detected  -> Extracts 63 features, runs inference normally.
    - 2 hands detected -> Reports 2 hands detected without error/crash. Safe guard prevents
                         sending 126 features to the 63-feature model.
  - Future model (126 features):
    - 1 or 2 hands detected -> Extracts deterministic 126-feature handedness vector
      ([Left: 63, Right: 63]) and runs inference.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parents[1]
_MODEL_PATH    = _BASE / "model" / "sign_language_model_wristrel_scale_17class.pkl"
_SCALER_PATH   = _BASE / "model" / "scaler_wristrel_scale_17class.pkl"
_ENCODER_PATH  = _BASE / "model" / "label_encoder_wristrel_scale_17class.pkl"
_METADATA_PATH = _BASE / "model" / "model_metadata_wristrel_scale_17class.json"

# ---------------------------------------------------------------------------
# Confidence threshold — predictions below this are treated as "unknown"
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = 0.45

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_model    = None
_scaler   = None
_encoder  = None
_metadata = None
_load_attempted = False
_load_success   = False


def _load_artifacts() -> bool:
    """Load the wrist-relative 17-class model, scaler, label encoder, and metadata from disk.

    Returns True on success, False on any error.
    Does NOT fall back to legacy model on failure.
    """
    global _model, _scaler, _encoder, _metadata, _load_attempted, _load_success
    if _load_attempted:
        return _load_success
    _load_attempted = True

    try:
        import joblib
    except ImportError:
        logger.error("joblib not installed. Run: pip install joblib")
        return False

    missing = [p for p in (_MODEL_PATH, _SCALER_PATH, _ENCODER_PATH) if not p.exists()]
    if missing:
        logger.error(
            "Missing wrist-relative 17-class model file(s): %s. Train the wristrel model first.",
            [str(p) for p in missing],
        )
        return False

    try:
        _model   = joblib.load(_MODEL_PATH)
        _scaler  = joblib.load(_SCALER_PATH)
        _encoder = joblib.load(_ENCODER_PATH)

        if _METADATA_PATH.exists():
            import json
            with open(_METADATA_PATH, "r", encoding="utf-8") as f:
                _metadata = json.load(f)

        _load_success = True
        logger.info(
            "Wrist-Relative 17-Class Model Loaded successfully (%d Features | %d Classes).",
            get_model_feature_dim(),
            len(get_supported_classes()),
        )
        return True
    except Exception as exc:
        logger.error("Failed to load wrist-relative 17-class model artifacts: %s", exc)
        return False


def get_model_status_info() -> Dict[str, Any]:
    """Return model metadata and load status for UI and backend diagnostics."""
    if not _load_attempted:
        _load_artifacts()
    dim = get_model_feature_dim()
    classes = get_supported_classes()
    return {
        "status_title": "My Personal 17-Class Model",
        "status_badge": f"{dim} Features | {len(classes)} Classes",
        "model_file": _MODEL_PATH.name,
        "feature_dim": dim,
        "num_classes": len(classes),
        "classes": classes,
        "model_loaded": _load_success,
    }


def is_model_loaded() -> bool:
    """Return True if all model artifacts are loaded and ready."""
    if not _load_attempted:
        _load_artifacts()
    return _load_success


def get_model_feature_dim() -> int:
    """Return the feature vector length expected by the loaded model (63 or 126)."""
    if not _load_success:
        _load_artifacts()

    if _scaler is not None and hasattr(_scaler, "n_features_in_"):
        return int(_scaler.n_features_in_)
    if _model is not None and hasattr(_model, "n_features_in_"):
        return int(_model.n_features_in_)
    return 63


def get_supported_classes() -> List[str]:
    """Return the list of class labels known to the loaded model."""
    if not _load_success:
        _load_artifacts()
    if _encoder is None:
        return []
    return list(_encoder.classes_)


def predict(
    hand_landmarks_list: list,
    handedness_list: Optional[list] = None,
) -> Tuple[Optional[str], float]:
    """Run sign recognition with adaptive 1-hand/2-hand compatibility.

    Args:
        hand_landmarks_list: List of detected hand landmark objects.
        handedness_list: Optional list of handedness dictionaries.

    Returns:
        (label, confidence) where:
            label      — predicted sign string (e.g. "HELLO"), or None if
                         prediction fails or confidence is below threshold.
            confidence — float in [0.0, 1.0].
    """
    res = predict_detailed(hand_landmarks_list, handedness_list)
    return res.get("label"), res.get("confidence", 0.0)


def predict_detailed(
    hand_landmarks_list: list,
    handedness_list: Optional[list] = None,
) -> Dict[str, Any]:
    """Detailed prediction interface returning status, metadata, and probabilities.

    Returns:
        Dictionary with keys:
            'label': Optional[str]
            'confidence': float
            'hands_detected': int
            'expected_dim': int
            'status': str ('success' | 'no_hands' | 'two_hands_ready_for_future_model' | 'low_confidence' | 'error')
            'message': str
    """
    if not _load_attempted:
        _load_artifacts()
    if not _load_success:
        return {
            "label": None,
            "confidence": 0.0,
            "hands_detected": 0,
            "expected_dim": 63,
            "status": "error",
            "message": "Model artifacts not loaded",
        }

    hand_count = len(hand_landmarks_list) if hand_landmarks_list else 0
    expected_dim = get_model_feature_dim()

    if hand_count == 0:
        return {
            "label": None,
            "confidence": 0.0,
            "hands_detected": 0,
            "expected_dim": expected_dim,
            "status": "no_hands",
            "message": "No hands detected",
        }

    # Case A: Current 1-Hand Model (expects 63 features)
    if expected_dim == 63:
        if hand_count >= 2:
            # Safe guard: Do NOT send 126 features to 63-feature model; report readiness
            return {
                "label": None,
                "confidence": 0.0,
                "hands_detected": hand_count,
                "expected_dim": 63,
                "status": "two_hands_ready_for_future_model",
                "two_hand_ready": True,
                "message": (
                    "Two hands detected — current model supports one-hand signs. "
                    "Two-hand recognition architecture is ready for future model retraining."
                ),
            }

        # Exactly 1 hand detected -> standard 63-feature extraction
        try:
            from modules.landmark_features import extract_single_hand_features
            features = extract_single_hand_features(hand_landmarks_list[0])
        except Exception as exc:
            logger.debug("Feature extraction failed: %s", exc)
            return {
                "label": None,
                "confidence": 0.0,
                "hands_detected": 1,
                "expected_dim": 63,
                "status": "error",
                "message": f"Feature extraction failed: {exc}",
            }

        if features is None or len(features) != 63:
            return {
                "label": None,
                "confidence": 0.0,
                "hands_detected": 1,
                "expected_dim": 63,
                "status": "error",
                "message": "Invalid feature length",
            }

    # Case B: Wrist-Relative 126-feature Model
    elif expected_dim == 126:
        try:
            from modules.landmark_features import extract_two_hand_wristrel
            features = extract_two_hand_wristrel(hand_landmarks_list, handedness_list)
        except Exception as exc:
            logger.debug("Wrist-relative feature extraction failed: %s", exc)
            return {
                "label": None,
                "confidence": 0.0,
                "hands_detected": hand_count,
                "expected_dim": 126,
                "status": "error",
                "message": f"Wrist-relative feature extraction failed: {exc}",
            }

        if features is None or len(features) != 126:
            return {
                "label": None,
                "confidence": 0.0,
                "hands_detected": hand_count,
                "expected_dim": 126,
                "status": "error",
                "message": "Invalid 126-feature length",
            }
    else:
        return {
            "label": None,
            "confidence": 0.0,
            "hands_detected": hand_count,
            "expected_dim": expected_dim,
            "status": "error",
            "message": f"Unexpected model feature dimension: {expected_dim}",
        }

    # Model Inference
    try:
        import numpy as np
        X = np.array(features, dtype=float).reshape(1, -1)
        X_scaled = _scaler.transform(X)

        y_encoded = _model.predict(X_scaled)
        proba = _model.predict_proba(X_scaled)
        confidence = float(proba[0].max())

        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "label": None,
                "confidence": confidence,
                "hands_detected": hand_count,
                "expected_dim": expected_dim,
                "status": "low_confidence",
                "message": "Confidence below threshold",
            }

        label = _encoder.inverse_transform(y_encoded)[0]
        return {
            "label": str(label),
            "confidence": confidence,
            "hands_detected": hand_count,
            "expected_dim": expected_dim,
            "status": "success",
            "message": "Prediction successful",
        }

    except Exception as exc:
        logger.debug("Model inference failed: %s", exc)
        return {
            "label": None,
            "confidence": 0.0,
            "hands_detected": hand_count,
            "expected_dim": expected_dim,
            "status": "error",
            "message": f"Inference failed: {exc}",
        }
