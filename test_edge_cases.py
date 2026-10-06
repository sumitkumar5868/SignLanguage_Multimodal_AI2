"""Edge case and error handling tests."""

import csv
from pathlib import Path
from modules.dataset_utils import (
    delete_last_sample,
    count_label_samples,
    validate_dataset,
    DATASET_PATH,
    build_header,
)


def test_edge_cases():
    """Test edge cases and error handling."""
    print("=" * 70)
    print(" EDGE CASE & ERROR HANDLING TESTS")
    print("=" * 70)
    print()
    
    # Test 1: Missing CSV file
    print("TEST 1: Delete from Non-Existent CSV")
    print("-" * 70)
    missing_path = DATASET_PATH.with_stem("sign_landmarks_missing")
    if missing_path.exists():
        missing_path.unlink()
    
    success, message = delete_last_sample("HELLO", path=missing_path)
    print(f"Result: {success}, Message: {message}")
    assert not success, "Should fail"
    assert message == "DATASET EMPTY", f"Expected 'DATASET EMPTY', got '{message}'"
    print("✓ Test 1 PASSED")
    print()
    
    # Test 2: Empty CSV (only header)
    print("TEST 2: Delete from Empty CSV (Only Header)")
    print("-" * 70)
    empty_path = DATASET_PATH.with_stem("sign_landmarks_empty")
    with open(empty_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(build_header())
    
    success, message = delete_last_sample("HELLO", path=empty_path)
    print(f"Result: {success}, Message: {message}")
    assert not success, "Should fail"
    assert message == "NO SAMPLE TO DELETE", f"Expected 'NO SAMPLE TO DELETE', got '{message}'"
    
    # Verify header still exists
    with open(empty_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    assert len(header) == 64, "Header was corrupted"
    print("✓ Header preserved")
    print("✓ Test 2 PASSED")
    empty_path.unlink()
    print()
    
    # Test 3: Delete from sign with no samples while other signs have samples
    print("TEST 3: Delete from Empty Sign (Other Signs Have Samples)")
    print("-" * 70)
    mixed_path = DATASET_PATH.with_stem("sign_landmarks_mixed")
    
    # Create CSV with only HELLO samples
    rows = [build_header()]
    for i in range(3):
        row = [str(0.1 + i * 0.001)] * 63 + ["HELLO"]
        rows.append(row)
    
    with open(mixed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    success, message = delete_last_sample("I_LOVE_YOU", path=mixed_path)
    print(f"Result: {success}, Message: {message}")
    assert not success, "Should fail"
    assert message == "NO SAMPLE TO DELETE", f"Expected 'NO SAMPLE TO DELETE', got '{message}'"
    
    # Verify HELLO samples still exist
    hello_count = count_label_samples("HELLO", path=mixed_path)
    assert hello_count == 3, "HELLO samples were affected"
    print("✓ Other signs' samples preserved")
    print("✓ Test 3 PASSED")
    mixed_path.unlink()
    print()
    
    # Test 4: Multiple deletions in sequence
    print("TEST 4: Multiple Deletions in Sequence")
    print("-" * 70)
    seq_path = DATASET_PATH.with_stem("sign_landmarks_sequence")
    
    # Create CSV with 5 HELLO samples
    rows = [build_header()]
    for i in range(5):
        row = [str(0.1 + i * 0.001)] * 63 + ["HELLO"]
        rows.append(row)
    
    with open(seq_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    initial_count = count_label_samples("HELLO", path=seq_path)
    print(f"Initial count: {initial_count}")
    
    for i in range(3):
        success, message = delete_last_sample("HELLO", path=seq_path)
        assert success, f"Delete {i+1} failed"
        count = count_label_samples("HELLO", path=seq_path)
        print(f"After delete {i+1}: {count} samples (Message: {message})")
    
    final_count = count_label_samples("HELLO", path=seq_path)
    assert final_count == 2, f"Expected 2 samples, got {final_count}"
    
    # Verify dataset still valid
    validation = validate_dataset(path=seq_path)
    assert validation['status'] == "VALID", "Dataset corrupted after multiple deletions"
    print("✓ Dataset remains valid after multiple deletions")
    print("✓ Test 4 PASSED")
    seq_path.unlink()
    print()
    
    # Test 5: Verify correct last sample is deleted (not first or random)
    print("TEST 5: Verify Correct Last Sample is Deleted")
    print("-" * 70)
    order_path = DATASET_PATH.with_stem("sign_landmarks_order")
    
    # Create CSV with identifiable samples (different feature values)
    rows = [build_header()]
    for i in range(3):
        # Make each sample unique by varying the first feature
        features = [str(0.1 + i * 0.1)] + [str(0.2)] * 62
        rows.append(features + ["HELLO"])
    
    with open(order_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    # Read last feature before deletion
    with open(order_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        last_first_feature = None
        for row in reader:
            last_first_feature = row[0]
    
    print(f"Last feature value before deletion: {last_first_feature}")
    
    # Delete last sample
    success, message = delete_last_sample("HELLO", path=order_path)
    assert success, "Delete failed"
    
    # Verify the truly last sample is now different
    with open(order_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        new_last_first_feature = None
        for row in reader:
            new_last_first_feature = row[0]
    
    print(f"Last feature value after deletion: {new_last_first_feature}")
    assert last_first_feature != new_last_first_feature, "Correct last sample was not deleted"
    print("✓ Correct last sample was deleted")
    print("✓ Test 5 PASSED")
    order_path.unlink()
    print()
    
    print("=" * 70)
    print(" ALL EDGE CASE TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_edge_cases()
