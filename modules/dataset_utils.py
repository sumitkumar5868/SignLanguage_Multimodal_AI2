"""Dataset utilities for Phase 5: landmark-based sign collection only."""

import csv
from pathlib import Path

FEATURE_COUNT = 63
LABEL_COLUMN = "label"
INITIAL_SIGNS = [
    "HELLO",
    "I_LOVE_YOU",
    "I_HATE_YOU",
    "I_EAT",
    "THANK_YOU",
]
TARGET_SAMPLES_PER_SIGN = 500
DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "sign_landmarks.csv"


def build_header():
    """Create the CSV header for 21 landmarks x 3 features + label."""
    header = []
    for index in range(21):
        header.extend([f"x{index}", f"y{index}", f"z{index}"])
    header.append(LABEL_COLUMN)
    return header


def ensure_dataset_file(path=None):
    """Create the dataset CSV with the correct header if it does not exist."""
    target_path = Path(path) if path else DATASET_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not target_path.exists():
        with open(target_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(build_header())

    return target_path


def append_sample(label, feature_values, path=None):
    """Append one sample row to the dataset file."""
    target_path = ensure_dataset_file(path)

    if feature_values is None or len(feature_values) != FEATURE_COUNT:
        raise ValueError(f"Invalid feature count: {len(feature_values) if feature_values is not None else 0}. Expected {FEATURE_COUNT}.")

    row = list(feature_values)
    row.append(label)

    with open(target_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(row)

    return row


def count_label_samples(label, path=None):
    """Count how many samples already exist for the selected label."""
    target_path = ensure_dataset_file(path)

    if not target_path.exists():
        return 0

    count = 0
    with open(target_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            if len(row) >= 64 and row[-1] == label:
                count += 1

    return count


def get_class_counts(path=None):
    """Return the count of rows per label."""
    target_path = ensure_dataset_file(path)
    counts = {label: 0 for label in INITIAL_SIGNS}

    with open(target_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for row in reader:
            if len(row) < 64:
                continue
            label = row[-1]
            if label in counts:
                counts[label] += 1

    return counts


def delete_last_sample(label, path=None):
    """Delete the most recently added sample for the given label with safety checks.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    target_path = Path(path) if path else DATASET_PATH
    
    # Handle missing CSV
    if not target_path.exists():
        return False, "DATASET EMPTY"
    
    try:
        # Read all rows
        rows = []
        with open(target_path, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            rows = list(reader)
        
        if len(rows) <= 1:  # Only header
            return False, "NO SAMPLE TO DELETE"
        
        # Find the last row with the matching label
        last_index = -1
        for i in range(len(rows) - 1, 0, -1):  # Start from end, skip header
            if len(rows[i]) >= 64 and rows[i][-1] == label:
                last_index = i
                break
        
        if last_index == -1:
            return False, "NO SAMPLE TO DELETE"
        
        # Create temporary backup
        backup_path = target_path.with_suffix(".backup")
        with open(backup_path, "w", newline="", encoding="utf-8") as backup_file:
            writer = csv.writer(backup_file)
            writer.writerows(rows)
        
        # Delete the found row
        rows.pop(last_index)
        
        # Write back to original file
        with open(target_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(rows)
        
        # Clean up backup
        backup_path.unlink(missing_ok=True)
        
        return True, "LAST SAMPLE DELETED"
    
    except Exception as exc:
        # Attempt to restore from backup if it exists
        backup_path = target_path.with_suffix(".backup")
        if backup_path.exists():
            try:
                backup_path.replace(target_path)
                return False, "DELETE FAILED — DATASET RESTORED"
            except Exception:
                return False, "DELETE FAILED — COULD NOT RESTORE"
        return False, f"DELETE FAILED — {str(exc)}"


def validate_dataset(path=None):
    """Validate dataset dimensions and row quality."""
    target_path = ensure_dataset_file(path)
    counts = get_class_counts(target_path)

    total_samples = 0
    missing_values = 0
    invalid_rows = 0
    expected_header = build_header()

    with open(target_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header is None:
            return {
                "total_samples": 0,
                "feature_columns": FEATURE_COUNT,
                "label_columns": 1,
                "labels": 5,
                "class_counts": counts,
                "missing_values": 0,
                "invalid_rows": 0,
                "status": "INVALID",
            }

        for row in reader:
            if not row:
                continue
            total_samples += 1
            if len(row) != 64:
                invalid_rows += 1
                continue

            features = row[:-1]
            label = row[-1]

            if any(value == "" for value in features):
                missing_values += 1
            if label not in counts:
                invalid_rows += 1

    valid = (
        header == expected_header
        and total_samples >= 0
        and invalid_rows == 0
        and missing_values == 0
    )

    return {
        "total_samples": total_samples,
        "feature_columns": FEATURE_COUNT,
        "label_columns": 1,
        "labels": len(INITIAL_SIGNS),
        "class_counts": counts,
        "missing_values": missing_values,
        "invalid_rows": invalid_rows,
        "status": "VALID" if valid else "INVALID",
    }
