"""Validate the Phase 5 dataset."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.dataset_utils import validate_dataset, ensure_dataset_file


def print_validation_report():
    """Print a formatted dataset validation report."""
    print("=" * 60)
    print(" DATASET VALIDATION REPORT")
    print("=" * 60)
    print()

    ensure_dataset_file()
    report = validate_dataset()

    dataset_path = Path(__file__).resolve().parents[1] / "dataset" / "sign_landmarks.csv"
    print(f"File: {dataset_path}")
    print()

    print(f"Total Samples: {report['total_samples']}")
    print(f"Features: {report['feature_columns']}")
    print(f"Label Column: label")
    print(f"Total Columns: {report['feature_columns'] + report['label_columns']}")
    print()

    print("Classes:")
    for label, count in report["class_counts"].items():
        print(f"  {label}: {count}")
    print()

    print(f"Missing Values: {report['missing_values']}")
    print(f"Invalid Rows: {report['invalid_rows']}")
    print()

    print(f"Dataset Status: {report['status']}")
    print()
    print("=" * 60)

    return report["status"] == "VALID"


if __name__ == "__main__":
    is_valid = print_validation_report()
    sys.exit(0 if is_valid else 1)
