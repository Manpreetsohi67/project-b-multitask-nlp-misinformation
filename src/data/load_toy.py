"""
src/data/load_toy.py
Module to load a tiny synthetic dataset for smoke testing
"""

import pandas as pd
from typing import Dict, List, Tuple
import random

def create_toy_dataset(num_samples: int = 20) -> pd.DataFrame:
    """
    Creates a tiny synthetic dataset for smoke testing with fields:
    text, event_id, sentiment_label, stance_label, veracity_label
    """
    # Sample texts related to various events
    texts = [
        "The new climate policy will help reduce emissions significantly.",
        "Experts confirm the effectiveness of the new vaccine.",
        "Scientists debate the impact of recent discoveries.",
        "Government announces new economic measures.",
        "Health officials warn about rising infection rates.",
        "Researchers publish findings on renewable energy.",
        "Social media platform updates its privacy policy.",
        "Study reveals benefits of regular exercise.",
        "Experts question the validity of the claims.",
        "New report shows positive economic growth.",
        "Climate activists demand immediate action.",
        "Technology company unveils innovative solution.",
        "Medical breakthrough offers hope for patients.",
        "Survey indicates growing public concern.",
        "Scientists express skepticism about results.",
        "Policy makers discuss urgent reforms.",
        "Educational institutions adapt to new norms.",
        "Industry leaders predict market trends.",
        "Public health experts recommend precautions.",
        "Environmental groups celebrate victory."
    ]
    
    # Event IDs to demonstrate event-holding strategy
    event_ids = [
        "climate_policy", "vaccine_study", "scientific_debate", 
        "economic_measures", "health_warnings", "energy_research",
        "privacy_policy", "exercise_study", "claim_validity", 
        "growth_report", "climate_action", "tech_solution",
        "medical_breakthrough", "public_concern", "scientific_skepticism",
        "policy_reforms", "educational_adaptation", "market_trends",
        "health_precautions", "environmental_victory"
    ]
    
    # Labels for the three tasks
    sentiment_labels = ["positive", "neutral", "negative"]
    stance_labels = ["support", "neutral", "oppose"]
    veracity_labels = ["true", "false", "uncertain"]
    
    # Create dataset
    data = []
    for i in range(min(num_samples, len(texts))):
        row = {
            "text": texts[i],
            "event_id": event_ids[i % len(event_ids)],
            "sentiment_label": random.choice(sentiment_labels),
            "stance_label": random.choice(stance_labels),
            "veracity_label": random.choice(veracity_labels)
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def load_toy_data() -> pd.DataFrame:
    """Load the toy dataset and print class counts"""
    df = create_toy_dataset()
    
    print("Loaded toy dataset with {} samples".format(len(df)))
    print("\nClass distribution:")
    print("\nSentiment labels:")
    print(df['sentiment_label'].value_counts())
    print("\nStance labels:")
    print(df['stance_label'].value_counts())
    print("\nVeracity labels:")
    print(df['veracity_label'].value_counts())
    
    return df

if __name__ == "__main__":
    # Smoke test
    df = load_toy_data()
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())