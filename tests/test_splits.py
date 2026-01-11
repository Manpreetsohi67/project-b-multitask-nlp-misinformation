"""
tests/test_splits.py
Test to verify event separation in train/test splits
"""

import pytest
import pandas as pd
from src.data.split_event_heldout import split_by_events, verify_event_separation
from src.data.load_toy import create_toy_dataset

def test_event_separation():
    """Test that train and test sets have no overlapping events"""
    # Create a toy dataset
    df = create_toy_dataset(num_samples=20)
    
    # Split by events
    train_df, test_df = split_by_events(df, test_size=0.3, random_state=42)
    
    # Verify no event overlap
    assert verify_event_separation(train_df, test_df), "Events should not overlap between train and test sets"
    
    # Verify that both sets have some data
    assert len(train_df) > 0, "Training set should not be empty"
    assert len(test_df) > 0, "Test set should not be empty"
    
    # Verify that all events in train are in the original dataset
    train_events = set(train_df['event_id'].unique())
    original_events = set(df['event_id'].unique())
    assert train_events.issubset(original_events), "Training events should be subset of original events"
    
    # Verify that all events in test are in the original dataset
    test_events = set(test_df['event_id'].unique())
    assert test_events.issubset(original_events), "Test events should be subset of original events"
    
    print(f"✓ Event separation test passed: {len(train_df)} train samples, {len(test_df)} test samples")
    print(f"✓ {len(train_events)} training events, {len(test_events)} test events")
    print(f"✓ No overlap between train and test events")

def test_split_sizes():
    """Test that split sizes are approximately correct"""
    # Create a larger dataset to test proportions
    df = create_toy_dataset(num_samples=30)  # Use more samples for better proportion testing
    
    # Split by events
    train_df, test_df = split_by_events(df, test_size=0.3, random_state=42)
    
    # Calculate event proportions (not sample proportions, since we split by events)
    total_events = len(df['event_id'].unique())
    train_events = len(train_df['event_id'].unique())
    test_events = len(test_df['event_id'].unique())
    
    # Check that we have reasonable split (allowing for rounding)
    expected_test_events = int(total_events * 0.3)
    expected_train_events = total_events - expected_test_events
    
    # Allow for some variation due to discrete nature of events
    assert abs(test_events - expected_test_events) <= 1, f"Test events {test_events} not close to expected {expected_test_events}"
    assert abs(train_events - expected_train_events) <= 1, f"Train events {train_events} not close to expected {expected_train_events}"
    
    print(f"✓ Split size test passed: {train_events}/{test_events} events (expected ~{expected_train_events}/{expected_test_events})")

def test_all_samples_preserved():
    """Test that all samples from original dataset are preserved in splits"""
    df = create_toy_dataset(num_samples=25)
    original_sample_count = len(df)
    original_indices = set(df.index)
    
    train_df, test_df = split_by_events(df, test_size=0.3, random_state=42)
    combined_sample_count = len(train_df) + len(test_df)
    
    # Verify all samples are preserved
    assert combined_sample_count == original_sample_count, f"Sample count mismatch: {combined_sample_count} vs {original_sample_count}"
    
    # Verify no duplicate samples between train and test
    train_indices = set(train_df.index)
    test_indices = set(test_df.index)
    assert len(train_indices.intersection(test_indices)) == 0, "No samples should appear in both train and test sets"
    
    # Verify all original indices are accounted for
    assert train_indices.union(test_indices) == original_indices, "All original samples should be in either train or test"
    
    print(f"✓ Sample preservation test passed: {original_sample_count} samples preserved across splits")

if __name__ == "__main__":
    print("Running event split tests...")
    test_event_separation()
    test_split_sizes()
    test_all_samples_preserved()
    print("\n✓ All event split tests passed!")