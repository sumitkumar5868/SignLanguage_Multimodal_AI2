"""
Switch the live app to use the new scale-normalized model.
Run this AFTER build_scale_model.py completes successfully.
"""
from pathlib import Path
import sys

ROOT = Path("/Users/rohitroy/Downloads/SignLanguage_Multimodal_AI")
MODEL_DIR = ROOT / "model"

REQUIRED = [
    MODEL_DIR / "sign_language_model_wristrel_scale_17class.pkl",
    MODEL_DIR / "scaler_wristrel_scale_17class.pkl",
    MODEL_DIR / "label_encoder_wristrel_scale_17class.pkl",
]
for p in REQUIRED:
    if not p.exists():
        print(f"ERROR: {p} not found. Run build_scale_model.py first.")
        sys.exit(1)

PREDICTOR = ROOT / "modules" / "predictor.py"
text = PREDICTOR.read_text()

# Update model file references
text = text.replace(
    '"sign_language_model_mydata.pkl"',
    '"sign_language_model_wristrel_scale_17class.pkl"'
).replace(
    '"scaler_mydata.pkl"',
    '"scaler_wristrel_scale_17class.pkl"'
).replace(
    '"label_encoder_mydata.pkl"',
    '"label_encoder_wristrel_scale_17class.pkl"'
).replace(
    '"model_metadata_mydata.json"',
    '"model_metadata_wristrel_scale_17class.json"'
)

PREDICTOR.write_text(text)
print("predictor.py updated to use scale-normalized model.")
print()
print("Restart the server:")
print("  venv/bin/python3 app.py --port 5001")
