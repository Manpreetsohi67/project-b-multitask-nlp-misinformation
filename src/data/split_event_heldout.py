"""
src/data/split_event_heldout.py
Module to split data by event_id for event-held-out evaluation
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import numpy as np

def split_by_events(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by event_id to ensure no overlap between train and test sets.
    This prevents data leakage in event-held-out evaluation.
    
    Args:
        df: DataFrame with 'event_id' column
        test_size: Proportion of events to use for test set
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, test_df: DataFrames split by unique events
    """
    # Get unique events
    unique_events = df['event_id'].unique()
    
    # Split events into train and test
    train_events, test_events = train_test_split(
        unique_events, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Create train and test sets based on event splits
    train_df = df[df['event_id'].isin(train_events)].copy()
    test_df = df[df['event_id'].isin(test_events)].copy()
    
    return train_df, test_df

def print_split_summary(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Print a summary of the event-held-out split"""
    print("Event-Held-Out Split Summary:")
    print("="*50)
    
    print(f"Training set: {len(train_df)} samples from {train_df['event_id'].nunique()} events")
    print(f"Test set: {len(test_df)} samples from {test_df['event_id'].nunique()} events")
    
    print(f"\nTraining events: {sorted(train_df['event_id'].unique())}")
    print(f"Test events: {sorted(test_df['event_id'].unique())}")
    
    # Verify no overlap
    train_events = set(train_df['event_id'].unique())
    test_events = set(test_df['event_id'].unique())
    overlap = train_events.intersection(test_events)
    
    print(f"\nEvent overlap check: {'PASS' if len(overlap) == 0 else 'FAIL'} (overlap: {overlap})")
    
    # Show class distribution in each set
    print("\nTraining set class distribution:")
    print("\nSentiment:")
    print(train_df['sentiment_label'].value_counts())
    print("\nStance:")
    print(train_df['stance_label'].value_counts())
    print("\nVeracity:")
    print(train_df['veracity_label'].value_counts())
    
    print("\nTest set class distribution:")
    print("\nSentiment:")
    print(test_df['sentiment_label'].value_counts())
    print("\nStance:")
    print(test_df['stance_label'].value_counts())
    print("\nVeracity:")
    print(test_df['veracity_label'].value_counts())

def verify_event_separation(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Verify that train and test sets have no overlapping events"""
    train_events = set(train_df['event_id'].unique())
    test_events = set(test_df['event_id'].unique())
    overlap = train_events.intersection(test_events)
    return len(overlap) == 0

if __name__ == "__main__":
    # Import and test with toy data
    from src.data.load_toy import load_toy_data
    
    print("Loading toy dataset...")
    df = load_toy_data()
    
    print(f"\nSplitting data by events...")
    train_df, test_df = split_by_events(df, test_size=0.3, random_state=42)
    
    print_split_summary(train_df, test_df)
    
    # Verify event separation
    is_separated = verify_event_separation(train_df, test_df)
    print(f"\nEvent separation verified: {is_separated}")