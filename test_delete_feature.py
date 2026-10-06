"""Test script for the delete last sample feature."""

import csv
from pathlib import Path
from modules.dataset_utils import (
    delete_last_sample,
    count_label_samples,
    validate_dataset,
    DATASET_PATH,
    build_header,
)


def create_test_csv():
    """Create a test CSV with known samples."""
    test_path = DATASET_PATH.with_stem("sign_landmarks_test")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create header and sample data
    header = build_header()
    test_data = [header]
    
    # Add 3 HELLO samples
    for i in range(3):
        row = [str(0.1 + i * 0.01)] * 63 + ["HELLO"]
        test_data.append(row)
    
    # Add 2 I_LOVE_YOU samples
    for i in range(2):
        row = [str(0.2 + i * 0.01)] * 63 + ["I_LOVE_YOU"]
        test_data.append(row)
    
    # Add 1 I_HATE_YOU sample
    row = [str(0.3)] * 63 + ["I_HATE_YOU"]
    test_data.append(row)
    
    # Write to test CSV
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(test_data)
    
    return test_path


def test_delete_feature():
    """Test the delete feature with a test CSV."""
    print("=" * 60)
    print(" TESTING DELETE LAST SAMPLE FEATURE")
    print("=" * 60)
    print()
    
    # Create test CSV
    test_path = create_test_csv()
    print(f"✓ Test CSV created: {test_path}")
    print()
    
    # Test 1: Check initial counts
    print("TEST 1: Initial Sample Counts")
    print("-" * 60)
    hello_count = count_label_samples("HELLO", path=test_path)
    i_love_you_count = count_label_samples("I_LOVE_YOU", path=test_path)
    i_hate_you_count = count_label_samples("I_HATE_YOU", path=test_path)
    
    print(f"HELLO: {hello_count} samples")
    print(f"I_LOVE_YOU: {i_love_you_count} samples")
    print(f"I_HATE_YOU: {i_hate_you_count} samples")
    
    assert hello_count == 3, f"Expected 3 HELLO samples, got {hello_count}"
    assert i_love_you_count == 2, f"Expected 2 I_LOVE_YOU samples, got {i_love_you_count}"
    assert i_hate_you_count == 1, f"Expected 1 I_HATE_YOU sample, got {i_hate_you_count}"
    print("✓ Test 1 PASSED")
    print()
    
    # Test 2: Delete HELLO sample
    print("TEST 2: Delete Last HELLO Sample")
    print("-" * 60)
    success, message = delete_last_sample("HELLO", path=test_path)
    print(f"Status: {success}")
    print(f"Message: {message}")
    
    assert success, f"Delete failed: {message}"
    assert message == "LAST SAMPLE DELETED", f"Unexpected message: {message}"
    
    hello_count = count_label_samples("HELLO", path=test_path)
    print(f"HELLO count after delete: {hello_count}")
    assert hello_count == 2, f"Expected 2 HELLO samples after delete, got {hello_count}"
    
    # Verify other counts unchanged
    i_love_you_count = count_label_samples("I_LOVE_YOU", path=test_path)
    i_hate_you_count = count_label_samples("I_HATE_YOU", path=test_path)
    assert i_love_you_count == 2, "I_LOVE_YOU count changed!"
    assert i_hate_you_count == 1, "I_HATE_YOU count changed!"
    print("✓ Test 2 PASSED")
    print()
    
    # Test 3: Delete I_LOVE_YOU sample
    print("TEST 3: Delete Last I_LOVE_YOU Sample")
    print("-" * 60)
    success, message = delete_last_sample("I_LOVE_YOU", path=test_path)
    print(f"Status: {success}")
    print(f"Message: {message}")
    
    assert success, f"Delete failed: {message}"
    i_love_you_count = count_label_samples("I_LOVE_YOU", path=test_path)
    print(f"I_LOVE_YOU count after delete: {i_love_you_count}")
    assert i_love_you_count == 1, f"Expected 1 I_LOVE_YOU sample after delete, got {i_love_you_count}"
    
    # Verify other counts unchanged
    hello_count = count_label_samples("HELLO", path=test_path)
    i_hate_you_count = count_label_samples("I_HATE_YOU", path=test_path)
    assert hello_count == 2, "HELLO count changed!"
    assert i_hate_you_count == 1, "I_HATE_YOU count changed!"
    print("✓ Test 3 PASSED")
    print()
    
    # Test 4: Delete all samples of a sign
    print("TEST 4: Delete All Samples of a Sign (I_HATE_YOU)")
    print("-" * 60)
    success, message = delete_last_sample("I_HATE_YOU", path=test_path)
    print(f"Status: {success}")
    print(f"Message: {message}")
    
    assert success, f"Delete failed: {message}"
    i_hate_you_count = count_label_samples("I_HATE_YOU", path=test_path)
    print(f"I_HATE_YOU count after delete: {i_hate_you_count}")
    assert i_hate_you_count == 0, f"Expected 0 I_HATE_YOU samples after delete, got {i_hate_you_count}"
    print("✓ Test 4 PASSED")
    print()
    
    # Test 5: Try to delete from empty sign
    print("TEST 5: Try to Delete from Empty Sign")
    print("-" * 60)
    success, message = delete_last_sample("I_EAT", path=test_path)
    print(f"Status: {success}")
    print(f"Message: {message}")
    
    assert not success, "Should fail when deleting from empty sign"
    assert message == "NO SAMPLE TO DELETE", f"Expected 'NO SAMPLE TO DELETE', got '{message}'"
    print("✓ Test 5 PASSED")
    print()
    
    # Test 6: Validate dataset
    print("TEST 6: Validate Dataset After Deletions")
    print("-" * 60)
    validation = validate_dataset(path=test_path)
    print(f"Total samples: {validation['total_samples']}")
    print(f"Feature columns: {validation['feature_columns']}")
    print(f"Label columns: {validation['label_columns']}")
    print(f"Class counts: {validation['class_counts']}")
    print(f"Missing values: {validation['missing_values']}")
    print(f"Invalid rows: {validation['invalid_rows']}")
    print(f"Status: {validation['status']}")
    
    assert validation['status'] == "VALID", f"Dataset validation failed: {validation['status']}"
    assert validation['total_samples'] == 3, f"Expected 3 total samples, got {validation['total_samples']}"
    assert validation['feature_columns'] == 63, "Feature count mismatch"
    assert validation['label_columns'] == 1, "Label column count mismatch"
    assert validation['missing_values'] == 0, "Dataset has missing values"
    assert validation['invalid_rows'] == 0, "Dataset has invalid rows"
    print("✓ Test 6 PASSED")
    print()
    
    # Test 7: Delete from non-existent CSV
    print("TEST 7: Try to Delete from Non-Existent CSV")
    print("-" * 60)
    non_existent_path = DATASET_PATH.with_stem("sign_landmarks_nonexistent")
    if non_existent_path.exists():
        non_existent_path.unlink()
    
    success, message = delete_last_sample("HELLO", path=non_existent_path)
    print(f"Status: {success}")
    print(f"Message: {message}")
    
    assert not success, "Should fail when CSV doesn't exist"
    assert message == "DATASET EMPTY", f"Expected 'DATASET EMPTY', got '{message}'"
    print("✓ Test 7 PASSED")
    print()
    
    # Clean up test CSV
    test_path.unlink()
    print("=" * 60)
    print(" ALL TESTS PASSED ✓")
    print("=" * 60)
    print()
    print("Summary:")
    print("- Delete function correctly removes last sample of current sign")
    print("- Other signs' samples are preserved")
    print("- CSV header is preserved")
    print("- Dataset validation passes after deletions")
    print("- Proper error handling for empty signs and missing CSV")


if __name__ == "__main__":
    test_delete_feature()
