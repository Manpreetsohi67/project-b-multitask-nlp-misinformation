"""
src/data/label_map.py
Module to demonstrate label harmonization mappings
"""

from typing import Dict, Any

def get_sentiment_mappings() -> Dict[str, str]:
    """
    Define mappings for sentiment label harmonization
    """
    # Example mapping from various sources to standardized labels
    mappings = {
        # From various sentiment scales to our 3-class system
        "pos": "positive",
        "neg": "negative", 
        "neu": "neutral",
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "very_positive": "positive",
        "very_negative": "negative",
        "somewhat_positive": "positive",
        "somewhat_negative": "negative",
        "mixed": "neutral",
        "conflicting": "neutral",
        # Numerical mappings
        "1": "negative",  # Assuming scale like 1-5 where 1=negative
        "2": "negative",
        "3": "neutral", 
        "4": "positive",
        "5": "positive",
        1: "negative",
        2: "negative", 
        3: "neutral",
        4: "positive",
        5: "positive"
    }
    return mappings

def get_stance_mappings() -> Dict[str, str]:
    """
    Define mappings for stance label harmonization
    """
    mappings = {
        # From various stance annotations to our 3-class system
        "in_favor": "support",
        "favor": "support",
        "supports": "support",
        "for": "support",
        "pro": "support",
        "agree": "support",
        
        "against": "oppose",
        "anti": "oppose",
        "disagree": "oppose",
        "counter": "oppose",
        "opposes": "oppose",
        
        "neutral": "neutral",
        "none": "neutral",
        "other": "neutral",
        "unknown": "neutral",
        "no_stance": "neutral",
        "not_sure": "neutral",
        
        # Numerical mappings
        "0": "neutral",
        "1": "support", 
        "-1": "oppose",
        0: "neutral",
        1: "support",
        -1: "oppose"
    }
    return mappings

def get_veracity_mappings() -> Dict[str, str]:
    """
    Define mappings for veracity label harmonization
    """
    mappings = {
        # From fact-checking labels to our 3-class system
        "true": "true",
        "truth": "true", 
        "correct": "true",
        "accurate": "true",
        "fact": "true",
        "real": "true",
        "verified_true": "true",
        
        "false": "false",
        "lie": "false",
        "incorrect": "false",
        "inaccurate": "false", 
        "fiction": "false",
        "hoax": "false",
        "fake": "false",
        "misleading": "false",  # Depending on interpretation
        "partially_false": "false",  # Simplified to false
        
        "uncertain": "uncertain",
        "unverified": "uncertain",
        "unconfirmed": "uncertain",
        "unknown": "uncertain",
        "unproven": "uncertain",
        "partially_true": "uncertain",  # Could be treated as uncertain
        "mixed": "uncertain",
        "needs_verification": "uncertain",
        
        # Numerical mappings
        "0": "uncertain",  # Unverified
        "1": "true",       # True
        "2": "false",      # False
        0: "uncertain",
        1: "true", 
        2: "false"
    }
    return mappings

def harmonize_labels(label: Any, task_type: str) -> str:
    """
    Harmonize a label to the standard format based on task type
    
    Args:
        label: Original label to harmonize
        task_type: One of 'sentiment', 'stance', 'veracity'
    
    Returns:
        Harmonized label string
    """
    label_str = str(label).lower().strip()
    
    if task_type == "sentiment":
        mappings = get_sentiment_mappings()
    elif task_type == "stance":
        mappings = get_stance_mappings()
    elif task_type == "veracity":
        mappings = get_veracity_mappings()
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return mappings.get(label_str, label_str)

def print_mapping_summary():
    """Print a summary of all label mappings for documentation"""
    print("Label Mapping Summary:")
    print("="*50)
    
    print("\nSentiment Mappings:")
    print(get_sentiment_mappings())
    
    print("\nStance Mappings:")
    print(get_stance_mappings())
    
    print("\nVeracity Mappings:")
    print(get_veracity_mappings())

if __name__ == "__main__":
    # Demonstrate label harmonization
    print("Demonstrating label harmonization...")
    
    # Test sentiment mappings
    test_sentiments = ["pos", "negative", "very_positive", 1, "neu"]
    print("\nSentiment harmonization:")
    for label in test_sentiments:
        harmonized = harmonize_labels(label, "sentiment")
        print(f"  {label} -> {harmonized}")
    
    # Test stance mappings
    test_stances = ["in_favor", "against", "no_stance", 1]
    print("\nStance harmonization:")
    for label in test_stances:
        harmonized = harmonize_labels(label, "stance")
        print(f"  {label} -> {harmonized}")
        
    # Test veracity mappings
    test_veracities = ["true", "hoax", "unverified", 1]
    print("\nVeracity harmonization:")
    for label in test_veracities:
        harmonized = harmonize_labels(label, "veracity")
        print(f"  {label} -> {harmonized}")
        
    print_mapping_summary()