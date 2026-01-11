"""
src/models/multitask_model.py
Multi-task NLP model for sentiment, stance, and veracity detection
Uses a shared encoder with separate classification heads
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple
import torch.nn.functional as F

class MultiTaskModel(nn.Module):
    """
    Multi-task model with shared encoder and separate classification heads
    for sentiment, stance, and veracity detection
    """
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 num_sentiment_classes: int = 3,  # positive, neutral, negative
                 num_stance_classes: int = 3,      # support, neutral, oppose
                 num_veracity_classes: int = 3,    # true, false, uncertain
                 dropout_rate: float = 0.1):
        super(MultiTaskModel, self).__init__()
        
        # Load pre-trained transformer as shared encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze encoder initially (can be unfrozen later for fine-tuning)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Get hidden size from encoder
        hidden_size = self.encoder.config.hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification heads for each task
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiment_classes)
        self.stance_classifier = nn.Linear(hidden_size, num_stance_classes)
        self.veracity_classifier = nn.Linear(hidden_size, num_veracity_classes)
        
        # Store number of classes for each task
        self.num_sentiment_classes = num_sentiment_classes
        self.num_stance_classes = num_stance_classes
        self.num_veracity_classes = num_veracity_classes
        
        # Initialize classification heads
        nn.init.xavier_uniform_(self.sentiment_classifier.weight)
        nn.init.xavier_uniform_(self.stance_classifier.weight)
        nn.init.xavier_uniform_(self.veracity_classifier.weight)
        
        # Task names for easy access
        self.task_names = ['sentiment', 'stance', 'veracity']
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask to ignore padding tokens
            
        Returns:
            Dictionary with logits for each task
        """
        # Pass through encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the pooled output (CLS token representation)
        pooled_output = outputs.last_hidden_state[:, 0]  # Take CLS token
        pooled_output = self.dropout(pooled_output)
        
        # Pass through each classification head
        sentiment_logits = self.sentiment_classifier(pooled_output)
        stance_logits = self.stance_classifier(pooled_output)
        veracity_logits = self.veracity_classifier(pooled_output)
        
        return {
            'sentiment': sentiment_logits,
            'stance': stance_logits,
            'veracity': veracity_logits
        }
    
    def enable_encoder_grads(self):
        """Enable gradients for encoder parameters (for fine-tuning)"""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def disable_encoder_grads(self):
        """Disable gradients for encoder parameters (freeze encoder)"""
        for param in self.encoder.parameters():
            param.requires_grad = False

def get_model_summary(model: MultiTaskModel) -> Dict:
    """Get a summary of the model architecture"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'encoder_frozen': not any(p.requires_grad for p in model.encoder.parameters()),
        'num_sentiment_classes': model.num_sentiment_classes,
        'num_stance_classes': model.num_stance_classes,
        'num_veracity_classes': model.num_veracity_classes
    }

def print_model_summary(model: MultiTaskModel):
    """Print a formatted summary of the model"""
    summary = get_model_summary(model)
    
    print("Multi-Task Model Architecture:")
    print("="*50)
    print(f"Total parameters: {summary['total_parameters']:,}")
    print(f"Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"Encoder frozen: {summary['encoder_frozen']}")
    print(f"Sentiment classes: {summary['num_sentiment_classes']}")
    print(f"Stance classes: {summary['num_stance_classes']}")
    print(f"Veracity classes: {summary['num_veracity_classes']}")
    
    print("\nModel components:")
    print("- Shared encoder: DistilBERT")
    print("- Sentiment classifier: Linear layer")
    print("- Stance classifier: Linear layer") 
    print("- Veracity classifier: Linear layer")

# Example usage and smoke test
if __name__ == "__main__":
    print("Initializing multi-task model...")
    
    # Create model instance
    model = MultiTaskModel()
    
    # Print model summary
    print_model_summary(model)
    
    # Create dummy inputs for testing forward pass
    print("\nTesting forward pass...")
    
    # Use a simple tokenizer for testing
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Sample text
    texts = ["This is a sample text for testing.", "Another example sentence."]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Forward pass
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    
    print(f"\nForward pass successful!")
    print(f"Output shapes:")
    for task, logits in outputs.items():
        print(f"  {task}: {logits.shape}")
    
    # Check if outputs have expected dimensions
    expected_batch_size = len(texts)
    assert outputs['sentiment'].shape[0] == expected_batch_size
    assert outputs['stance'].shape[0] == expected_batch_size
    assert outputs['veracity'].shape[0] == expected_batch_size
    print(f"\nOutput dimensions verified for batch size {expected_batch_size}")