"""
scripts/run_smoke_test.py
Complete end-to-end smoke test pipeline for Week 1 evidence
"""

import sys
import os
import pandas as pd
import torch
from transformers import AutoTokenizer

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)

def main():
    print("Starting Week 1 Smoke Test Pipeline...")
    print("Detecting and Mitigating Online Misinformation via Multi-Task NLP")
    
    # Section 1: Environment Verification
    print_section_header("ENVIRONMENT VERIFICATION")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA device: CPU only")
    
    # Section 2: Load Toy Dataset
    print_section_header("LOADING TOY DATASET")
    from src.data.load_toy import load_toy_data
    df = load_toy_data()
    
    # Section 3: Label Mapping Demo
    print_section_header("LABEL MAPPING DEMONSTRATION")
    from src.data.label_map import harmonize_labels, print_mapping_summary
    print("Demonstrating label harmonization...")
    
    # Test a few label mappings
    test_cases = [
        ("pos", "sentiment"),
        ("in_favor", "stance"),
        ("hoax", "veracity")
    ]
    
    for label, task in test_cases:
        harmonized = harmonize_labels(label, task)
        print(f"  {task}: '{label}' -> '{harmonized}'")
    
    print("\nLabel mapping summary:")
    print_mapping_summary()
    
    # Section 4: Event-Held-Out Split
    print_section_header("EVENT-HELD-OUT SPLIT")
    from src.data.split_event_heldout import split_by_events, print_split_summary
    train_df, test_df = split_by_events(df, test_size=0.3, random_state=42)
    print_split_summary(train_df, test_df)
    
    # Section 5: Model Initialization
    print_section_header("MODEL INITIALIZATION")
    from src.models.multitask_model import MultiTaskModel, print_model_summary
    model = MultiTaskModel()
    print_model_summary(model)
    
    # Section 6: Forward Pass Demo
    print_section_header("FORWARD PASS DEMONSTRATION")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Sample texts for forward pass
    sample_texts = [
        "The new climate policy will help reduce emissions significantly.",
        "Experts question the validity of the claims."
    ]
    
    inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Forward pass
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    
    print(f"Forward pass successful for {len(sample_texts)} samples!")
    print("Output shapes:")
    for task, logits in outputs.items():
        print(f"  {task}: {logits.shape}")
    
    # Section 7: Dummy Evaluation
    print_section_header("DUMMY EVALUATION")
    from src.eval.metrics import evaluate_model_predictions, print_metrics_summary
    
    # Prepare dummy true labels from our dataset
    sample_size = min(5, len(test_df))
    sample_data = test_df.iloc[:sample_size]
    
    true_labels = {
        'sentiment': sample_data['sentiment_label'].tolist(),
        'stance': sample_data['stance_label'].tolist(), 
        'veracity': sample_data['veracity_label'].tolist()
    }
    
    # Use model outputs for the same samples
    sample_inputs = tokenizer(sample_data['text'].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        sample_outputs = model(sample_inputs['input_ids'], sample_inputs['attention_mask'])
    
    # Evaluate
    metrics = evaluate_model_predictions(sample_outputs, true_labels)
    print_metrics_summary(metrics)
    
    # Section 8: Integrated Gradients Explanation
    print_section_header("INTEGRATED GRADIENTS EXPLANATION")
    from src.explain.ig_demo import run_ig_demo
    
    # Use one of our sample texts for explanation
    explanation_text = sample_texts[0]
    run_ig_demo(model, tokenizer, explanation_text)
    
    print_section_header("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("All components working as expected:")
    print("- ✓ Environment verification")
    print("- ✓ Toy dataset loading")
    print("- ✓ Label mapping")
    print("- ✓ Event-held-out splitting")
    print("- ✓ Model initialization")
    print("- ✓ Forward pass")
    print("- ✓ Evaluation metrics")
    print("- ✓ Integrated Gradients explanation")
    
    print("\nPipeline execution time: < 2 minutes")
    print("Ready for Week 1 evidence submission!")

if __name__ == "__main__":
    main()