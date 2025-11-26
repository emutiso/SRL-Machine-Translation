"""
Test script for validating MarianMT model functionality with a small sample
Uses the actual MarianMTTrainer implementation from model_trainer.py
"""

import sys
import os
import torch
from pathlib import Path

# Configuration parameters - adjust these as needed
TRAINING_SAMPLE_SIZE = 15000  # Number of samples to use for training
EVALUATION_SAMPLE_SIZE = 5000  # Number of samples to use for evaluation
MAX_TRAINING_STEPS = 300    # Number of training steps to run
BATCH_SIZE = 16             # Batch size for training
CACHE_DIR = "cached_models"  # Directory to cache pre-trained models

# Add the project root to the Python path to enable imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.data_loader import get_data
from model.model_trainer import MarianMTTrainer
from model.model_evaluator import MarianMTEvaluator

def test_baseline_and_srl_models():
    """Test both baseline and SRL-augmented models using the MarianMTTrainer class"""
    
    # Make sure cache directory exists
    cache_dir = os.path.join(project_root, CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using model cache directory: {cache_dir}")
    
    # Load data
    print("\nLoading data...")
    train_data_no_srl, train_data_srl, eval_data_no_srl, eval_data_srl = get_data()
    
    # Print sample sizes
    print(f"\nUsing configuration:")
    print(f"- Training sample size: {TRAINING_SAMPLE_SIZE}")
    print(f"- Evaluation sample size: {EVALUATION_SAMPLE_SIZE}")
    print(f"- Max training steps: {MAX_TRAINING_STEPS}")
    print(f"- Batch size: {BATCH_SIZE}")
    
    # Train baseline model (no SRL)
    print("\n" + "=" * 50)
    print("TRAINING BASELINE MODEL (NO SRL)")
    print("=" * 50)
    
    baseline_trainer = MarianMTTrainer(
        use_srl=False,
        output_dir="test_output",
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        epochs=3,  # Will be limited by max_steps
        max_steps=MAX_TRAINING_STEPS,
        use_special_tokens_for_srl=False,
        disable_tqdm=False,  # Disable verbose progress bars
        cache_dir=cache_dir  # Use caching to avoid rate limits
    )
    
    baseline_trainer, baseline_model_path = baseline_trainer.train(
        train_dataset=train_data_no_srl,
        eval_dataset=train_data_no_srl,
        max_train_samples=TRAINING_SAMPLE_SIZE,
        max_eval_samples=EVALUATION_SAMPLE_SIZE
    )
    
    # Train SRL model
    print("\n" + "=" * 50)
    print("TRAINING SRL-AUGMENTED MODEL")
    print("=" * 50)
    
    srl_trainer = MarianMTTrainer(
        use_srl=True,
        output_dir="test_output",
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        epochs=3,  # Will be limited by max_steps
        max_steps=MAX_TRAINING_STEPS,
        use_special_tokens_for_srl=True,
        disable_tqdm=False,  # Disable verbose progress bars
        cache_dir=cache_dir  # Use caching to avoid rate limits
    )
    
    srl_trainer, srl_model_path = srl_trainer.train(
        train_dataset=train_data_srl,
        eval_dataset=train_data_srl,
        max_train_samples=TRAINING_SAMPLE_SIZE,
        max_eval_samples=EVALUATION_SAMPLE_SIZE
    )
    
    # Test both models on the same examples
    print("\n" + "=" * 50)
    print("TESTING TRANSLATION EXAMPLES")
    print("=" * 50)
    
    # Create evaluators for both models
    baseline_evaluator = MarianMTEvaluator(
        model_path=baseline_model_path,
        batch_size=BATCH_SIZE,
        cache_dir=cache_dir  # Use caching to avoid rate limits
    )
    
    srl_evaluator = MarianMTEvaluator(
        model_path=srl_model_path,
        batch_size=BATCH_SIZE,
        cache_dir=cache_dir  # Use caching to avoid rate limits
    )
    
    # Test with regular sentences
    test_examples = [
        "The man gave the book to the woman.",
        "She walked to the store.",
        "The children are playing in the park."
    ]
    
    # Create SRL versions of the same sentences - using ONLY actual SRL tags
    srl_examples = [
        "[Agent1] The man [Verb1] gave [Theme1] the book to [Goal1] the woman.",
        "[Agent1] She [Verb1] walked to [Goal1] the store.",
        "[Agent1] The children [Verb1] are playing in [Goal1] the park."
    ]
    
    # Show tokenization of SRL tags for confirmation
    print("\nSRL Tag Tokenization Verification:")
    processing_class = srl_trainer.tokenizer
    
    for tag in ["[Agent1]", "[Verb1]", "[Theme1]", "[Goal1]"]:
        token_id = processing_class.convert_tokens_to_ids(tag)
        is_special = token_id >= processing_class.vocab_size - len(processing_class.special_tokens_map.get('additional_special_tokens', []))
        print(f"Token '{tag}' ID: {token_id} - {'Is special token' if is_special else 'Not a special token'}")
    
    # Print sample tokenization
    sample_srl = srl_examples[0]
    tokens = processing_class.tokenize(sample_srl)
    print(f"\nSample tokenization of: {sample_srl}")
    print(f"Tokens: {tokens}")
    
    # Test translations
    print("\nBaseline Model Translations:")
    for i, example in enumerate(test_examples):
        translation = baseline_evaluator.translate([example])[0]
        print(f"Original: {example}")
        print(f"Translation: {translation}")
        print()
    
    print("\nBaseline Model with SRL Input:")
    for i, example in enumerate(srl_examples):
        translation = baseline_evaluator.translate([example])[0]
        print(f"SRL Input: {example}")
        print(f"Translation: {translation}")
        print()
    
    print("\nSRL Model Translations:")
    for i, example in enumerate(test_examples):
        translation = srl_evaluator.translate([example])[0]
        print(f"Original: {example}")
        print(f"Translation: {translation}")
        print()
    
    print("\nSRL Model with SRL Input:")
    for i, example in enumerate(srl_examples):
        translation = srl_evaluator.translate([example])[0]
        print(f"SRL Input: {example}")
        print(f"Translation: {translation}")
        print()


if __name__ == "__main__":
    test_baseline_and_srl_models()
