import os
import torch
import random
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from typing import Dict, List, Union, Tuple


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
        model_path: str,
        device: str = None
) -> Tuple[MarianMTModel, MarianTokenizer]:
    """Load a MarianMT model and tokenizer"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    model = model.to(device)
    return model, tokenizer


def compare_translations(
        source_texts: List[str],
        reference_texts: List[str],
        baseline_translations: List[str],
        srl_translations: List[str],
        n_examples: int = 5
) -> str:
    """Format a comparison of translations for display"""
    n_examples = min(n_examples, len(source_texts))
    comparison = []

    for i in range(n_examples):
        comparison.append(f"Example {i + 1}:")
        comparison.append(f"Source: {source_texts[i]}")
        comparison.append(f"Reference: {reference_texts[i]}")
        comparison.append(f"Baseline: {baseline_translations[i]}")
        comparison.append(f"SRL-augmented: {srl_translations[i]}")
        comparison.append("")

    return "\n".join(comparison)