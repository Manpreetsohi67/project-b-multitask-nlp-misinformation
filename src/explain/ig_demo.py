"""
src/explain/ig_demo.py
Integrated Gradients explanation demo for multi-task model
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from captum.attr import IntegratedGradients
from typing import Dict, List, Tuple
import numpy as np

class IGExplainer:
    """
    Class to run Integrated Gradients explanations on the multi-task model
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.ig = IntegratedGradients(self.model)
        
        # Define label mappings for converting indices back to labels
        self.idx_to_sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.idx_to_stance = {0: 'oppose', 1: 'neutral', 2: 'support'}
        self.idx_to_veracity = {0: 'false', 1: 'uncertain', 2: 'true'}
        
        self.label_mappings = {
            'sentiment': self.idx_to_sentiment,
            'stance': self.idx_to_stance,
            'veracity': self.idx_to_veracity
        }
    
    def get_word_attributions(self, text: str, task: str, target_class_idx: int = None) -> Tuple[List[str], List[float]]:
        """
        Get token-level attributions for a given text and task using Integrated Gradients
        
        Args:
            text: Input text to explain
            task: Task name ('sentiment', 'stance', or 'veracity')
            target_class_idx: Index of the target class to explain (if None, uses predicted class)
            
        Returns:
            Tuple of (tokens, attributions)
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Set model to eval mode
        self.model.eval()
        
        # Get model prediction if target class not specified
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            if target_class_idx is None:
                # Use the predicted class with highest logit
                task_logits = outputs[task]
                target_class_idx = torch.argmax(task_logits[0]).item()
        
        # Define custom forward function for IG
        def custom_forward(input_ids, attention_mask):
            outputs = self.model(input_ids, attention_mask)
            return outputs[task]
        
        # Apply Integrated Gradients
        attributions, delta = self.ig.attribute(
            inputs=(input_ids, attention_mask),
            baselines=(torch.zeros_like(input_ids), torch.zeros_like(attention_mask)),
            target=target_class_idx,
            additional_forward_args=(attention_mask,),
            return_convergence_delta=True
        )
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get attribution scores for the first (and only) example in the batch
        attr_scores = attributions[0].detach().numpy()
        
        # Only return attributions for actual tokens (not padding)
        actual_token_count = attention_mask.sum().item()
        tokens = tokens[:actual_token_count]
        attr_scores = attr_scores[:actual_token_count]
        
        return tokens, attr_scores.tolist()
    
    def get_top_attributed_tokens(self, text: str, task: str, target_class_idx: int = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k tokens with highest attribution scores for a specific task
        
        Args:
            text: Input text to explain
            task: Task name ('sentiment', 'stance', or 'veracity')
            target_class_idx: Index of the target class to explain (if None, uses predicted class)
            top_k: Number of top tokens to return
            
        Returns:
            List of (token, attribution_score) tuples
        """
        tokens, attributions = self.get_word_attributions(text, task, target_class_idx)
        
        # Create pairs and sort by absolute attribution values (most important regardless of sign)
        token_attr_pairs = [(token, attr) for token, attr in zip(tokens, attributions)]
        token_attr_pairs = sorted(token_attr_pairs, key=lambda x: abs(x[1]), reverse=True)
        
        # Return top-k
        return token_attr_pairs[:top_k]
    
    def explain_prediction(self, text: str, task: str, target_class_idx: int = None) -> Dict:
        """
        Comprehensive explanation of a prediction for a given task
        
        Args:
            text: Input text to explain
            task: Task name ('sentiment', 'stance', or 'veracity')
            target_class_idx: Index of the target class to explain (if None, uses predicted class)
            
        Returns:
            Dictionary with explanation details
        """
        # Get top attributed tokens
        top_tokens = self.get_top_attributed_tokens(text, task, target_class_idx, top_k=10)
        
        # Get model prediction
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            task_logits = outputs[task][0]  # First example in batch
            predicted_class_idx = torch.argmax(task_logits).item()
            predicted_class = self.label_mappings[task][predicted_class_idx]
            
            if target_class_idx is None:
                target_class_idx = predicted_class_idx
            target_class = self.label_mappings[task][target_class_idx]
        
        explanation = {
            'input_text': text,
            'task': task,
            'predicted_class': predicted_class,
            'target_class': target_class,
            'top_attributed_tokens': top_tokens,
            'all_tokens_with_attrs': list(zip(*self.get_word_attributions(text, task, target_class_idx)))
        }
        
        return explanation

def print_explanation_summary(explanation: Dict):
    """
    Print a formatted summary of the IG explanation
    """
    print("Integrated Gradients Explanation:")
    print("="*60)
    print(f"Input text: {explanation['input_text']}")
    print(f"Task: {explanation['task']}")
    print(f"Predicted class: {explanation['predicted_class']}")
    print(f"Target class: {explanation['target_class']}")
    
    print(f"\nTop 5 tokens contributing to {explanation['target_class']} prediction:")
    print("-"*50)
    for i, (token, attr) in enumerate(explanation['top_attributed_tokens'][:5]):
        print(f"  {i+1}. '{token}' : {attr:.4f}")

def run_ig_demo(model, tokenizer, sample_text: str = None):
    """
    Run a demonstration of Integrated Gradients explanation
    """
    if sample_text is None:
        sample_text = "The new climate policy will help reduce emissions significantly."
    
    print("Running Integrated Gradients explanation demo...")
    
    # Initialize explainer
    explainer = IGExplainer(model, tokenizer)
    
    # Explain for each task
    tasks = ['sentiment', 'stance', 'veracity']
    
    for task in tasks:
        print(f"\n--- Explaining {task.upper()} prediction ---")
        explanation = explainer.explain_prediction(sample_text, task)
        print_explanation_summary(explanation)
        print()

# Example usage and smoke test
if __name__ == "__main__":
    print("Testing Integrated Gradients explanation...")
    
    # Import model and tokenizer
    from src.models.multitask_model import MultiTaskModel
    from transformers import AutoTokenizer
    
    # Load model and tokenizer
    model = MultiTaskModel()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Run demo
    sample_text = "The new climate policy will help reduce emissions significantly."
    run_ig_demo(model, tokenizer, sample_text)
    
    print("\nIntegrated Gradients demo completed successfully!")