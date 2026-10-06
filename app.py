"""Sign Language Multimodal AI — Flask Web Application Backend.

Runs the Real-Time Sign Language Recognition Web Application with Multilingual Text & Voice Output.

Supported Languages:
    - English 🇬🇧
    - Hindi 🇮🇳
    - Bhojpuri 🇮🇳
    - Odia 🇮🇳
    - Telugu 🇮🇳
    - Bengali 🇮🇳

Usage:
    python app.py
    (Then open http://127.0.0.1:5000 in your browser)
"""

import argparse
import base64
import io
import logging
import os
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_cors import CORS

from modules.audio import is_audio_available, speak_text
from modules.camera import run_camera
from modules.hand_detection import (
    detect_hands_fast,
    initialize_hands,
    process_hand_frame,
    process_hand_frame_with_handedness,
)
from modules.predictor import (
    _load_artifacts,
    get_model_feature_dim,
    get_supported_classes,
    is_model_loaded,
    predict,
    predict_detailed,
)
from modules.sign_map import (
    SUPPORTED_LANGUAGES,
    get_all_translations,
    get_supported_languages_list,
    get_translation,
)
from modules.tts_provider import tts_manager

# ---------------------------------------------------------------------------
# Logging & Server Config
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SignLanguageWebAI")

app = Flask(__name__)
CORS(app)

_hands_landmarker = None


def get_hand_landmarker():
    """Lazy-initialize and return the MediaPipe HandLandmarker instance."""
    global _hands_landmarker
    if _hands_landmarker is None:
        try:
            _hands_landmarker = initialize_hands(
                min_detection_confidence=0.15,
                min_presence_confidence=0.15,
            )
            logger.info("MediaPipe HandLandmarker initialized (confidence=0.15 — high sensitivity).")
        except Exception as exc:
            logger.error("Failed to initialize MediaPipe HandLandmarker: %s", exc)
    return _hands_landmarker


# ---------------------------------------------------------------------------
# Web Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main web application UI."""
    return render_template("index.html")


@app.route("/api/languages", methods=["GET"])
def api_languages():
    """Return list of supported languages with TTS capabilities metadata."""
    languages = []
    for lang_dict in get_supported_languages_list():
        lang_id = lang_dict["id"]
        tts_info = tts_manager.get_language_status(lang_id)
        combined = dict(lang_dict)
        combined["tts_info"] = tts_info
        languages.append(combined)

    return jsonify({
        "status": "success",
        "languages": languages,
    })


@app.route("/api/status", methods=["GET"])
def api_status():
    """Return backend server, ML model, and languages readiness."""
    _load_artifacts()
    classes = get_supported_classes()
    dim = get_model_feature_dim()
    loaded = is_model_loaded()

    return jsonify({
        "status": "online",
        "model_loaded": loaded,
        "model_title": "My Personal 17-Class Model" if loaded else "Model Load Failed",
        "model_badge": f"{dim} Features | {len(classes)} Classes" if loaded else "Unavailable",
        "model_file": "sign_language_model_mydata.pkl",
        "feature_dim": dim,
        "classes": classes,
        "languages": get_supported_languages_list(),
        "audio_available": is_audio_available(),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Receive base64 frame from browser, run MediaPipe + ML prediction, return JSON with all translations."""
    try:
        data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        image_data = data.get("image")

        if not image_data:
            return jsonify({"status": "error", "message": "No image data provided"}), 400

        # Strip Data URL header
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        # Decode base64 image bytes into OpenCV frame
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Failed to decode frame"}), 400

        # Flip frame horizontally to match dataset collection convention
        frame = cv2.flip(frame, 1)

        # Ensure minimum resolution for reliable MediaPipe hand detection at any distance
        h, w = frame.shape[:2]
        if w < 640 or h < 480:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

        # Hand Detection with MediaPipe
        start_detect = time.time()
        landmarker = get_hand_landmarker()
        if landmarker is None:
            return jsonify({"status": "error", "message": "Hand landmarker unavailable"}), 503

        # Process frame with ultra-fast handedness detection (no drawing overhead)
        detected_hands, handedness_list = detect_hands_fast(frame, landmarker)
        detection_ms = round((time.time() - start_detect) * 1000, 1)
        hand_count = len(detected_hands)


        if hand_count == 0:
            return jsonify({
                "status": "success",
                "hands_detected": 0,
                "detection_ms": detection_ms,
                "hand_status": "0 HANDS (Searching)",
                "feature_count": 126,
                "active_features": 0,
                "label": None,
                "confidence": 0.0,
                "translations": get_all_translations(None),
                "two_hand_ready": False,
                "message": "No hands detected",
            })

        # Run model inference on detected hand(s)
        pred_result = predict_detailed(detected_hands, handedness_list)
        label = pred_result.get("label")
        confidence = pred_result.get("confidence", 0.0)
        two_hand_ready = pred_result.get("two_hand_ready", False)
        message = pred_result.get("message", "")
        translations = get_all_translations(label)

        logger.info(
            "Diagnostic: %d hand(s) detected in %.1fms | 126 features | Pred: %s (conf: %.2f)",
            hand_count,
            detection_ms,
            label or "None",
            confidence,
        )

        active_feat_count = 63 if hand_count == 1 else 126

        return jsonify({
            "status": "success",
            "hands_detected": hand_count,
            "detection_ms": detection_ms,
            "feature_count": 126,
            "active_features": active_feat_count,
            "hand_status": "1 HAND" if hand_count == 1 else f"{hand_count} HANDS",
            "label": label,
            "confidence": float(round(confidence, 3)),
            "translations": translations,
            "two_hand_ready": two_hand_ready,
            "message": message,
        })

    except Exception as exc:
        logger.exception("Prediction API error: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/speak", methods=["POST"])
def api_speak():
    """Trigger server-side multilingual text-to-speech audio (non-blocking)."""
    try:
        data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        text = data.get("text", "").strip()
        language = data.get("language", "hindi").strip().lower()

        if not text or text == "—" or "उपलब्ध नहीं" in text:
            return jsonify({"status": "error", "message": "No valid text to speak"}), 400

        success, msg = speak_text(text, language=language)
        if not success:
            return jsonify({"status": "unsupported_audio", "message": msg}), 200

        return jsonify({"status": "success", "message": msg})
    except Exception as exc:
        logger.error("Audio speak error: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/tts", methods=["GET"])
def api_tts_stream():
    """Generate audio stream via the modular TTS Provider Pipeline (with caching)."""
    text = request.args.get("text", "").strip()
    language = request.args.get("lang", "hindi").strip().lower()

    if not text or text == "—":
        return Response(status=204)

    audio_bytes, message, provider = tts_manager.synthesize(text, language)

    if not audio_bytes:
        return jsonify({
            "error": "unsupported_tts",
            "message": message,
            "language": language,
        }), 404

    mp3_fp = io.BytesIO(audio_bytes)
    return send_file(
        mp3_fp,
        mimetype="audio/mpeg",
        as_attachment=False,
        download_name=f"{language}_speech.mp3",
    )


# ---------------------------------------------------------------------------
# Main Launcher
# ---------------------------------------------------------------------------

def main():
    """Start the Sign Language Multimodal AI web server."""
    parser = argparse.ArgumentParser(
        description="Sign Language Multimodal AI — Real-Time Multilingual Sign Recognition Web App"
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Launch the desktop dataset collection camera tool instead of the web application",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port number to bind to (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask in debug mode",
    )
    args = parser.parse_args()

    if args.collect:
        print("Starting Dataset Collection Camera Tool…")
        return run_camera()

    # Pre-load ML model artifacts
    loaded = _load_artifacts()
    classes = get_supported_classes()
    dim = get_model_feature_dim()

    print("=" * 65)
    print("  SIGN LANGUAGE MULTIMODAL AI")
    print("  Real-Time Sign Recognition with Multilingual Voice Output")
    print("=" * 65)
    print()
    if loaded:
        print(f"  OK 17-Class Model Loaded")
        print(f"  OK {dim} Features | {len(classes)} Classes")
        print(f"  OK Supported Signs ({len(classes)}): {', '.join(classes)}")
    else:
        print("  ERROR: Failed to load 17-class model artifacts!")
    print("  OK Supported Languages: English, Hindi, Bhojpuri, Odia, Telugu, Bengali")
    print("  OK Modular TTS Pipeline: Microsoft Edge Neural + Google TTS + Regional Adapter")
    print()
    print(f"  WEB APP URL: http://{args.host}:{args.port}")
    print("  Open the URL in Google Chrome, Brave, or Safari.")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 65)
    print()

    bind_port = args.port
    try:
        app.run(
            host=args.host,
            port=bind_port,
            debug=args.debug,
            threaded=True,
        )
    except OSError as err:
        if "Address already in use" in str(err) and bind_port == 5000:
            bind_port = 5001
            print(f"\n  WARNING: Port 5000 in use. Switched to: http://{args.host}:{bind_port}")
            app.run(
                host=args.host,
                port=bind_port,
                debug=args.debug,
                threaded=True,
            )
        else:
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
