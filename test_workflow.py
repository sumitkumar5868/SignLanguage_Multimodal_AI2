"""Integration test simulating the exact workflow from requirements."""

import csv
from pathlib import Path
from modules.dataset_utils import (
    delete_last_sample,
    count_label_samples,
    validate_dataset,
    DATASET_PATH,
    build_header,
    append_sample,
)


def simulate_workflow():
    """Simulate the exact workflow described in requirements."""
    print("=" * 70)
    print(" INTEGRATION TEST: PHASE 5 DATASET DELETION WORKFLOW")
    print("=" * 70)
    print()
    
    # Create a fresh test dataset
    test_path = DATASET_PATH.with_stem("sign_landmarks_workflow")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize empty CSV
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(build_header())
    
    print("STEP 1: Select HELLO")
    print("-" * 70)
    current_sign = "HELLO"
    print(f"Current Sign: {current_sign}")
    print()
    
    print("STEP 2: Capture 3 samples")
    print("-" * 70)
    for i in range(3):
        features = [str(0.1 + i * 0.001)] * 63
        append_sample(current_sign, features, path=test_path)
        count = count_label_samples(current_sign, path=test_path)
        print(f"Sample {i+1} captured → {current_sign} = {count}")
    
    hello_count = count_label_samples(current_sign, path=test_path)
    print(f"\nVerify: {current_sign} = {hello_count}")
    assert hello_count == 3, f"Expected 3, got {hello_count}"
    print("✓ VERIFIED")
    print()
    
    print("STEP 3: Capture one intentionally wrong sample")
    print("-" * 70)
    features = [str(0.2)] * 63
    append_sample(current_sign, features, path=test_path)
    hello_count = count_label_samples(current_sign, path=test_path)
    print(f"Wrong sample captured → {current_sign} = {hello_count}")
    assert hello_count == 4, f"Expected 4, got {hello_count}"
    print("✓ VERIFIED")
    print()
    
    print("STEP 4: Press BACKSPACE to delete the wrong sample")
    print("-" * 70)
    success, message = delete_last_sample(current_sign, path=test_path)
    print(f"Delete status: {success}")
    print(f"Delete message: {message}")
    assert success, f"Delete failed: {message}"
    assert message == "LAST SAMPLE DELETED", f"Unexpected message: {message}"
    
    hello_count = count_label_samples(current_sign, path=test_path)
    print(f"After deletion: {current_sign} = {hello_count}")
    assert hello_count == 3, f"Expected 3, got {hello_count}"
    print("✓ VERIFIED")
    print()
    
    print("STEP 5: Select I_LOVE_YOU")
    print("-" * 70)
    current_sign = "I_LOVE_YOU"
    print(f"Current Sign: {current_sign}")
    
    # Add some samples for I_LOVE_YOU
    for i in range(2):
        features = [str(0.3 + i * 0.001)] * 63
        append_sample(current_sign, features, path=test_path)
    
    ilove_count = count_label_samples(current_sign, path=test_path)
    print(f"I_LOVE_YOU = {ilove_count}")
    
    # Verify HELLO count is unchanged
    hello_count = count_label_samples("HELLO", path=test_path)
    print(f"HELLO = {hello_count} (should be unchanged)")
    assert hello_count == 3, "HELLO count was modified!"
    print("✓ VERIFIED - I_LOVE_YOU count is independent")
    print()
    
    print("STEP 6: Validate the CSV")
    print("-" * 70)
    validation = validate_dataset(path=test_path)
    
    print(f"Total samples: {validation['total_samples']}")
    print(f"Feature columns: {validation['feature_columns']}")
    print(f"Label columns: {validation['label_columns']}")
    print(f"Status: {validation['status']}")
    print()
    
    print("Class Counts:")
    for label, count in validation['class_counts'].items():
        if count > 0:
            print(f"  {label}: {count}")
    
    print()
    assert validation['status'] == "VALID", f"Dataset validation failed: {validation['status']}"
    assert validation['feature_columns'] == 63, "Feature count mismatch"
    assert validation['label_columns'] == 1, "Label column count mismatch"
    assert validation['missing_values'] == 0, "Dataset has missing values"
    assert validation['invalid_rows'] == 0, "Dataset has invalid rows"
    print("Expected: Dataset Status: VALID")
    print("✓ VERIFIED")
    print()
    
    # Clean up
    test_path.unlink()
    
    print("=" * 70)
    print(" WORKFLOW TEST COMPLETED SUCCESSFULLY ✓")
    print("=" * 70)
    print()
    print("Test Results:")
    print("✓ Successfully selected sign (HELLO)")
    print("✓ Successfully captured samples (3 samples)")
    print("✓ Successfully verified count (HELLO = 3)")
    print("✓ Successfully captured wrong sample (HELLO = 4)")
    print("✓ Successfully deleted last sample (HELLO = 3)")
    print("✓ Successfully selected different sign (I_LOVE_YOU)")
    print("✓ Successfully verified counts are independent")
    print("✓ Successfully validated dataset (VALID)")
    print()
    print("Dataset Integrity:")
    print("✓ CSV header preserved")
    print("✓ 63 feature columns + 1 label column = 64 columns")
    print("✓ No missing rows or corrupted data")
    print("✓ All samples for all signs preserved correctly")


if __name__ == "__main__":
    simulate_workflow()
