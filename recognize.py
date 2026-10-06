"""Sign Language Multimodal AI — Recognition Entry Point.

Launch the professional real-time sign recognition UI:
    python recognize.py

This is separate from app.py (dataset collection tool) which is preserved
completely unchanged.

Pipeline:
    Camera → MediaPipe Hand Landmarks → 63-Feature Vector →
    RandomForest Model → Sign Label → English + Hindi Display + Audio
"""

import logging
import sys
import tkinter as tk

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Verify required packages are importable before launching the UI."""
    missing = []

    deps = [
        ("cv2",        "opencv-python"),
        ("mediapipe",  "mediapipe"),
        ("PIL",        "Pillow"),
        ("numpy",      "numpy"),
        ("joblib",     "joblib"),
        ("sklearn",    "scikit-learn"),
    ]
    for module_name, package_name in deps:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        logger.error(
            "Missing required packages: %s\n"
            "Install them with:\n  pip install %s",
            missing, " ".join(missing),
        )
        return False

    # Optional but important
    optionals = [
        ("gtts",   "gtts",   "Hindi TTS"),
        ("pygame", "pygame", "Audio playback"),
    ]
    for module_name, package_name, feature in optionals:
        try:
            __import__(module_name)
        except ImportError:
            logger.warning(
                "Optional package '%s' not found — %s will be unavailable. "
                "Install with: pip install %s",
                package_name, feature, package_name,
            )

    return True


def main() -> int:
    """Application entry point."""
    print("=" * 60)
    print("  Sign Language Multimodal AI")
    print("  Real-Time Recognition with Hindi Voice Output")
    print("=" * 60)
    print()

    if not check_dependencies():
        return 1

    # Import UI after dependency check so import errors are handled gracefully
    try:
        from modules.ui import SignLanguageApp
    except Exception as exc:
        logger.error("Failed to import UI module: %s", exc)
        return 1

    # Pre-load model (logs any missing file errors before the window opens)
    try:
        from modules.predictor import _load_artifacts, is_model_loaded
        _load_artifacts()
        if is_model_loaded():
            from modules.predictor import get_supported_classes
            classes = get_supported_classes()
            print(f"✓ Model loaded  —  {len(classes)} signs: {', '.join(classes)}")
        else:
            print("⚠  Model could not be loaded. Recognition will be unavailable.")
            print("   Run: python -m modules.train_model  to retrain.")
    except Exception as exc:
        logger.warning("Model pre-load warning: %s", exc)

    print()
    print("Starting UI…")
    print()

    root = tk.Tk()

    # DPI scaling hint (macOS / HiDPI)
    try:
        root.tk.call("tk", "scaling", 1.0)
    except Exception:
        pass

    app = SignLanguageApp(root)

    # Window geometry — centred on screen
    window_w, window_h = 1060, 620
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = max(0, (screen_w - window_w) // 2)
    y = max(0, (screen_h - window_h) // 2)
    root.geometry(f"{window_w}x{window_h}+{x}+{y}")

    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
