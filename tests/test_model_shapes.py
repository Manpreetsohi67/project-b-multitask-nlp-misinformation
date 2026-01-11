"""
tests/test_model_shapes.py
Test to verify model output dimensions are correct
"""

import pytest
import torch
from transformers import AutoTokenizer
from src.models.multitask_model import MultiTaskModel

def test_model_output_shapes():
    """Test that model outputs have correct dimensions"""
    # Initialize model
    model = MultiTaskModel()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Create sample inputs
    sample_texts = [
        "This is a sample text for testing.",
        "Another example sentence for shape verification."
    ]
    
    inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    batch_size = len(sample_texts)
    
    # Check that outputs have correct shapes
    assert 'sentiment' in outputs, "Output should contain sentiment logits"
    assert 'stance' in outputs, "Output should contain stance logits" 
    assert 'veracity' in outputs, "Output should contain veracity logits"
    
    # Check dimensions for each task
    sentiment_shape = outputs['sentiment'].shape
    stance_shape = outputs['stance'].shape
    veracity_shape = outputs['veracity'].shape
    
    # Each should be [batch_size, num_classes]
    assert sentiment_shape[0] == batch_size, f"Batch dimension mismatch for sentiment: {sentiment_shape[0]} vs {batch_size}"
    assert stance_shape[0] == batch_size, f"Batch dimension mismatch for stance: {stance_shape[0]} vs {batch_size}"
    assert veracity_shape[0] == batch_size, f"Batch dimension mismatch for veracity: {veracity_shape[0]} vs {batch_size}"
    
    # Check number of classes
    assert sentiment_shape[1] == 3, f"Sentiment should have 3 classes, got {sentiment_shape[1]}"
    assert stance_shape[1] == 3, f"Stance should have 3 classes, got {stance_shape[1]}"
    assert veracity_shape[1] == 3, f"Veracity should have 3 classes, got {veracity_shape[1]}"
    
    print(f"✓ Model output shapes test passed:")
    print(f"  Sentiment: {sentiment_shape}")
    print(f"  Stance: {stance_shape}")
    print(f"  Veracity: {veracity_shape}")

def test_single_sample():
    """Test model with a single sample"""
    model = MultiTaskModel()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Single sample
    sample_text = "Single sample for testing."
    inputs = tokenizer([sample_text], return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    model.eval()
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    
    # Check shapes for single sample
    for task, logits in outputs.items():
        assert logits.shape[0] == 1, f"Single sample should have batch size 1, got {logits.shape[0]} for {task}"
        assert logits.shape[1] == 3, f"Each task should have 3 classes, got {logits.shape[1]} for {task}"
    
    print("✓ Single sample test passed")

def test_model_parameters():
    """Test model parameter counts and structure"""
    model = MultiTaskModel()
    
    # Get parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Encoder parameters should be frozen by default
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    classifier_params = sum(p.numel() for p in [
        *model.sentiment_classifier.parameters(),
        *model.stance_classifier.parameters(), 
        *model.veracity_classifier.parameters()
    ])
    
    # Verify structure
    assert total_params == encoder_params + classifier_params, "Total params should equal encoder + classifier params"
    assert trainable_params == classifier_params, "Only classifier params should be trainable initially"
    
    # Enable encoder gradients and verify
    model.enable_encoder_grads()
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_after > trainable_params, "Trainable params should increase after enabling encoder gradients"
    
    # Disable encoder gradients and verify
    model.disable_encoder_grads()
    trainable_final = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_final == classifier_params, "Trainable params should return to classifier-only after disabling encoder gradients"
    
    print(f"✓ Model parameters test passed:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params (frozen encoder): {trainable_params:,}")
    print(f"  Encoder params: {encoder_params:,}")
    print(f"  Classifier params: {classifier_params:,}")

def test_different_batch_sizes():
    """Test model with different batch sizes"""
    model = MultiTaskModel()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        # Create batch of texts
        sample_texts = [f"Sample text {i} for batch size testing." for i in range(batch_size)]
        inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        model.eval()
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
        
        # Verify all outputs have correct batch size
        for task, logits in outputs.items():
            assert logits.shape[0] == batch_size, f"Batch size mismatch for {task}: {logits.shape[0]} vs {batch_size}"
            assert logits.shape[1] == 3, f"Class count mismatch for {task}: {logits.shape[1]} vs 3"
    
    print(f"✓ Different batch sizes test passed for sizes: {batch_sizes}")

if __name__ == "__main__":
    print("Running model shape tests...")
    test_model_output_shapes()
    test_single_sample()
    test_model_parameters()
    test_different_batch_sizes()
    print("\n✓ All model shape tests passed!")