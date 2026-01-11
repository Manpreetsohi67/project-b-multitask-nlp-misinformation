"""
src/eval/metrics.py
Evaluation metrics for multi-task learning: macro-F1 per task
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import Dict, List, Union, Tuple
import warnings

def compute_metrics_for_task(y_true: List[int], y_pred: List[int], task_name: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for a single task
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        task_name: Name of the task for reporting
    
    Returns:
        Dictionary with computed metrics
    """
    # Convert to numpy arrays for sklearn compatibility
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_np, y_pred_np)
    
    # Calculate macro F1 (average across all classes)
    f1_macro = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    
    # Calculate micro F1 (global average)
    f1_micro = f1_score(y_true_np, y_pred_np, average='micro', zero_division=0)
    
    # Calculate precision and recall (macro average)
    precision_macro = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    recall_macro = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    
    return {
        f'{task_name}_accuracy': accuracy,
        f'{task_name}_f1_macro': f1_macro,
        f'{task_name}_f1_micro': f1_micro,
        f'{task_name}_precision_macro': precision_macro,
        f'{task_name}_recall_macro': recall_macro
    }

def compute_multitask_metrics(
    true_labels: Dict[str, List[int]], 
    pred_labels: Dict[str, List[int]]
) -> Dict[str, float]:
    """
    Compute evaluation metrics for all tasks in the multi-task model
    
    Args:
        true_labels: Dictionary with ground truth labels for each task
        pred_labels: Dictionary with predicted labels for each task
    
    Returns:
        Dictionary with metrics for all tasks
    """
    all_metrics = {}
    
    # Compute metrics for each task
    for task_name in true_labels.keys():
        if task_name in pred_labels:
            task_metrics = compute_metrics_for_task(
                true_labels[task_name], 
                pred_labels[task_name], 
                task_name
            )
            all_metrics.update(task_metrics)
    
    return all_metrics

def print_metrics_summary(metrics: Dict[str, float]):
    """
    Print a formatted summary of the evaluation metrics
    """
    print("Evaluation Metrics Summary:")
    print("="*50)
    
    # Group metrics by task
    tasks = set()
    for key in metrics.keys():
        task = key.split('_')[0]
        tasks.add(task)
    
    for task in sorted(tasks):
        print(f"\n{task.upper()} Metrics:")
        print("-" * 30)
        for key, value in metrics.items():
            if key.startswith(task + '_'):
                metric_name = key.replace(task + '_', '').replace('_', ' ').title()
                print(f"  {metric_name}: {value:.4f}")

def get_predictions_from_logits(logits: torch.Tensor) -> List[int]:
    """
    Convert model logits to predicted class indices
    
    Args:
        logits: Model output logits (batch_size, num_classes)
    
    Returns:
        List of predicted class indices
    """
    _, predicted_indices = torch.max(logits, dim=1)
    return predicted_indices.tolist()

def convert_labels_to_indices(labels: List[str], label_to_idx: Dict[str, int]) -> List[int]:
    """
    Convert string labels to integer indices based on mapping
    
    Args:
        labels: List of string labels
        label_to_idx: Dictionary mapping string labels to indices
    
    Returns:
        List of integer indices
    """
    return [label_to_idx[label] for label in labels]

def get_label_mappings():
    """
    Get label to index mappings for each task
    """
    sentiment_to_idx = {'negative': 0, 'neutral': 1, 'positive': 2}
    stance_to_idx = {'oppose': 0, 'neutral': 1, 'support': 2}
    veracity_to_idx = {'false': 0, 'uncertain': 1, 'true': 2}
    
    return {
        'sentiment': sentiment_to_idx,
        'stance': stance_to_idx,
        'veracity': veracity_to_idx
    }

def evaluate_model_predictions(model_outputs: Dict[str, torch.Tensor], 
                             true_labels: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Full evaluation pipeline: convert logits to predictions, map string labels to indices, compute metrics
    
    Args:
        model_outputs: Dictionary with model logits for each task
        true_labels: Dictionary with true string labels for each task
    
    Returns:
        Dictionary with computed metrics
    """
    # Get label mappings
    label_mappings = get_label_mappings()
    
    # Convert logits to predictions
    pred_labels = {}
    for task, logits in model_outputs.items():
        pred_labels[task] = get_predictions_from_logits(logits)
    
    # Convert true labels to indices
    true_labels_idx = {}
    for task, labels in true_labels.items():
        true_labels_idx[task] = convert_labels_to_indices(labels, label_mappings[task])
    
    # Verify that predictions and true labels have the same length
    for task in pred_labels.keys():
        assert len(pred_labels[task]) == len(true_labels_idx[task]), \
            f"Mismatch in {task}: {len(pred_labels[task])} predictions vs {len(true_labels_idx[task])} labels"
    
    # Compute metrics
    metrics = compute_multitask_metrics(true_labels_idx, pred_labels)
    
    return metrics

# Example usage and smoke test
if __name__ == "__main__":
    print("Testing metrics computation...")
    
    # Simulate some dummy predictions and true labels
    batch_size = 10
    num_classes = 3
    
    # Dummy true labels (as strings for realistic scenario)
    true_sentiment = ['positive', 'negative', 'neutral', 'positive', 'negative', 
                      'neutral', 'positive', 'neutral', 'negative', 'positive']
    true_stance = ['support', 'oppose', 'neutral', 'support', 'oppose', 
                   'neutral', 'support', 'neutral', 'oppose', 'support']
    true_veracity = ['true', 'false', 'uncertain', 'true', 'false', 
                     'uncertain', 'true', 'uncertain', 'false', 'true']
    
    # Dummy logits (simulating model outputs)
    sentiment_logits = torch.randn(batch_size, num_classes)
    stance_logits = torch.randn(batch_size, num_classes)
    veracity_logits = torch.randn(batch_size, num_classes)
    
    # Organize into required format
    model_outputs = {
        'sentiment': sentiment_logits,
        'stance': stance_logits,
        'veracity': veracity_logits
    }
    
    true_labels = {
        'sentiment': true_sentiment,
        'stance': true_stance,
        'veracity': true_veracity
    }
    
    # Evaluate
    metrics = evaluate_model_predictions(model_outputs, true_labels)
    
    # Print results
    print_metrics_summary(metrics)
    
    print(f"\nSuccessfully computed metrics for {len(metrics)} different metrics!")