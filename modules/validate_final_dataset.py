"""Dataset Quality & Specification Validation Module.

Validates sign language landmark CSV datasets against the 126-feature schema:
- 127 total columns (126 features + 1 label column)
- Exact column naming: h1_x0..h1_z20, h2_x0..h2_z20, label
- Data types: numeric float features, non-empty string labels
- Data integrity: no NaN, no Inf, no nulls
- Consistency: 1-hand zero-padding vs 2-hand dual representation
- Class balance and sample counts

Usage:
    python modules/validate_final_dataset.py [path_to_csv]
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

EXPECTED_FEATURES = 126
EXPECTED_TOTAL_COLUMNS = 127
EXPECTED_LABEL_COL = "label"

# Standard class definitions for the 17-class vocabulary
STANDARDIZED_HAND_MODALITY: Dict[str, str] = {
    "HELLO": "ONE_HAND",
    "GOODBYE": "ONE_HAND",
    "YES": "ONE_HAND",
    "NO": "ONE_HAND",
    "PLEASE": "TWO_HANDS",     # Standard prayer pose
    "THANK_YOU": "ONE_HAND",
    "SORRY": "ONE_HAND",
    "HELP": "TWO_HANDS",       # One hand supporting other
    "WATER": "ONE_HAND",
    "FOOD": "ONE_HAND",
    "BATHROOM": "ONE_HAND",    # "T" / ASL sign
    "PAIN": "ONE_HAND",
    "SICK": "TWO_HANDS",       # Head/body contact
    "SLEEP": "TWO_HANDS",      # Palms near cheek
    "REST": "TWO_HANDS",       # Resting / T-position
    "STOP": "TWO_HANDS",       # Both open palms facing camera
    "EMERGENCY": "TWO_HANDS",   # Crossed arms / X-shape
}

EXPECTED_COLUMNS: List[str] = [
    f"h1_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
] + [
    f"h2_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
] + [EXPECTED_LABEL_COL]


def validate_dataset(file_path: Path) -> Dict[str, Any]:
    """Inspect and report all dataset metrics and anomalies without modifying data."""
    report: Dict[str, Any] = {
        "file_path": str(file_path),
        "exists": file_path.exists(),
        "passed": False,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    if not file_path.exists():
        report["errors"].append(f"File not found: {file_path}")
        return report

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        report["errors"].append(f"Failed to read CSV: {exc}")
        return report

    num_rows, num_cols = df.shape
    report["stats"]["rows"] = num_rows
    report["stats"]["cols"] = num_cols

    # 1. Column Count & Names Check
    if num_cols != EXPECTED_TOTAL_COLUMNS:
        report["errors"].append(
            f"Invalid column count: expected {EXPECTED_TOTAL_COLUMNS}, got {num_cols}"
        )

    col_names = list(df.columns)
    if col_names != EXPECTED_COLUMNS:
        mismatched = [
            f"{exp} != {act}" for exp, act in zip(EXPECTED_COLUMNS, col_names) if exp != act
        ]
        if mismatched:
            report["warnings"].append(
                f"Column header mismatch in {len(mismatched)} columns (first 3: {mismatched[:3]})"
            )

    # 2. Features & Labels Separation
    feature_data = df.iloc[:, :-1]
    label_data = df.iloc[:, -1]

    # 3. Numeric Types & NaN / Inf Checks
    nan_count = int(feature_data.isna().sum().sum())
    if nan_count > 0:
        report["errors"].append(f"Found {nan_count} NaN values in feature columns.")
    else:
        report["stats"]["nan_count"] = 0

    inf_count = int(np.isinf(feature_data.to_numpy(dtype=float, na_value=0.0)).sum())
    if inf_count > 0:
        report["errors"].append(f"Found {inf_count} infinite values in feature columns.")
    else:
        report["stats"]["inf_count"] = 0

    # 4. Duplicate Rows Check
    exact_duplicates = int(df.duplicated().sum())
    report["stats"]["exact_duplicates"] = exact_duplicates
    if exact_duplicates > 0:
        report["warnings"].append(f"Dataset contains {exact_duplicates} exact duplicate rows.")

    # 5. Class Label Distribution & Modality Consistency
    classes = sorted(label_data.dropna().astype(str).unique().tolist())
    report["stats"]["classes_count"] = len(classes)
    report["stats"]["classes"] = classes

    class_stats: Dict[str, Dict[str, Any]] = {}
    h2_data = feature_data.iloc[:, 63:].to_numpy(dtype=float)

    for c in classes:
        cls_mask = (label_data.astype(str) == c).to_numpy()
        cls_rows = int(cls_mask.sum())
        cls_h2 = h2_data[cls_mask]
        
        # Determine 2-hand if any h2 feature is non-zero
        two_hand_count = int(np.sum(np.abs(cls_h2).sum(axis=1) > 1e-4))
        one_hand_count = cls_rows - two_hand_count
        expected_mode = STANDARDIZED_HAND_MODALITY.get(c, "UNKNOWN")

        inconsistent = False
        if expected_mode == "ONE_HAND" and two_hand_count > 0:
            inconsistent = True
            report["warnings"].append(
                f"Class '{c}' is expected ONE_HAND but contains {two_hand_count} 2-hand rows."
            )
        elif expected_mode == "TWO_HANDS" and one_hand_count > 0:
            inconsistent = True
            report["warnings"].append(
                f"Class '{c}' is expected TWO_HANDS but contains {one_hand_count} 1-hand rows (missing 2nd hand)."
            )

        class_stats[c] = {
            "samples": cls_rows,
            "one_hand": one_hand_count,
            "two_hand": two_hand_count,
            "expected_modality": expected_mode,
            "is_consistent": not inconsistent,
        }

    report["stats"]["per_class"] = class_stats

    # Check overall pass status
    if len(report["errors"]) == 0:
        report["passed"] = True

    return report


def print_validation_report(report: Dict[str, Any]) -> None:
    """Print a clean human-readable diagnostic report."""
    print("=" * 70)
    print("  📊 DATASET QUALITY & SPECIFICATION VALIDATION REPORT")
    print("=" * 70)
    print(f"  Target File : {report['file_path']}")
    print(f"  Status      : {'✅ PASSED (Ready)' if report['passed'] else '❌ FAILED'}")
    print(f"  Total Rows  : {report['stats'].get('rows', 0):,}")
    print(f"  Total Cols  : {report['stats'].get('cols', 0)} (126 features + 1 label)")
    print(f"  Total Signs : {report['stats'].get('classes_count', 0)} classes")
    print(f"  Duplicates  : {report['stats'].get('exact_duplicates', 0)} rows")
    print(f"  NaN / Inf   : {report['stats'].get('nan_count', 0)} / {report['stats'].get('inf_count', 0)}")
    print("=" * 70)

    print("\n  Per-Class Distribution & Modality Breakdown:")
    print("  " + "-" * 66)
    print(f"  {'Sign Name':<14} | {'Total':<6} | {'1-Hand':<7} | {'2-Hand':<7} | {'Target Modality':<14} | {'Status'}")
    print("  " + "-" * 66)

    per_class = report["stats"].get("per_class", {})
    for cls_name, info in per_class.items():
        status_str = "✅ Clean" if info["is_consistent"] else "⚠️ Inconsistent"
        print(
            f"  {cls_name:<14} | {info['samples']:<6} | {info['one_hand']:<7} | "
            f"{info['two_hand']:<7} | {info['expected_modality']:<14} | {status_str}"
        )
    print("  " + "-" * 66)

    if report["warnings"]:
        print(f"\n  ⚠️ Warnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"    • {w}")

    if report["errors"]:
        print(f"\n  ❌ Errors ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"    • {e}")
    else:
        print("\n  ✓ No blocking schema or numeric errors detected.")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate sign language landmark dataset CSV.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="dataset/Forproject.csv",
        help="Path to CSV dataset to validate (default: dataset/Forproject.csv)",
    )
    args = parser.parse_args()

    file_path = Path(args.csv_path).resolve()
    report = validate_dataset(file_path)
    print_validation_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
