"""
Model evaluator module for assessing MarianMT model performance
"""

import os
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import evaluate
from transformers import MarianMTModel, MarianTokenizer
from typing import Dict, List, Union, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = os.path.join(PROJECT_ROOT, "cached_models")


class MarianMTEvaluator:
    def __init__(
            self,
            model_path: str,
            max_length: int = 128,
            batch_size: int = 16,
            use_cuda: bool = True,
            cache_dir: str = CACHE_DIR
    ):
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.cache_dir = cache_dir

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load tokenizer and model (with caching if using a HF model)
        self.processing_class, self.model = self._load_model_and_tokenizer()
        self.model = self.model.to(self.device)

        # Load metrics
        self.bleu_metric = evaluate.load("sacrebleu")
        self.ter_metric = evaluate.load("ter")
        
        print(f"Evaluator initialized with model: {model_path}")
        print(f"Using device: {self.device}")
    
    def _get_cache_path(self):
        """Get path for cached model
        
        Only applies to HuggingFace models, not to local paths
        """
        # Check if this is a huggingface model path or a local path
        if "/" in self.model_path and not os.path.exists(self.model_path):
            # This looks like a HuggingFace model ID
            model_dir_name = self.model_path.replace("/", "_")
            return os.path.join(self.cache_dir, model_dir_name)
        return None  # Local path, no caching needed
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with caching for HuggingFace models"""
        cache_path = self._get_cache_path()
        
        # If it's a local path or an already fine-tuned model, load directly
        if cache_path is None:
            print(f"Loading model from local path: {self.model_path}")
            return MarianTokenizer.from_pretrained(self.model_path), MarianMTModel.from_pretrained(self.model_path)
        
        # Check if model is already cached
        if os.path.exists(cache_path):
            print(f"Loading model from local cache: {cache_path}")
            try:
                tokenizer = MarianTokenizer.from_pretrained(cache_path)
                model = MarianMTModel.from_pretrained(cache_path)
                print("Successfully loaded model from cache")
                return tokenizer, model
            except Exception as e:
                print(f"Error loading from cache: {e}")
                print("Downloading model from Hugging Face instead...")
                # If loading fails, delete the corrupted cache
                shutil.rmtree(cache_path, ignore_errors=True)
        
        # If no cache or cache loading failed, download from Hugging Face
        print(f"Downloading model from Hugging Face: {self.model_path}")
        tokenizer = MarianTokenizer.from_pretrained(self.model_path)
        model = MarianMTModel.from_pretrained(self.model_path)
        
        # Save to cache for future use
        print(f"Saving model to local cache: {cache_path}")
        tokenizer.save_pretrained(cache_path)
        model.save_pretrained(cache_path)
        
        return tokenizer, model

    def postprocess_text(self, preds, labels):
        """Clean up the generated text and references for evaluation"""
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(self, eval_preds):
        """Compute BLEU and TER metrics"""
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predictions and labels
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.processing_class.pad_token_id)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=True)

        # Post-process
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        # Compute metrics
        bleu_result = self.bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        ter_result = self.ter_metric.compute(predictions=decoded_preds, references=decoded_labels)

        result = {
            "bleu": bleu_result["score"],
            "ter": ter_result["score"]
        }

        # Add prediction length
        prediction_lens = [np.count_nonzero(pred != self.processing_class.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        # Round values
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    def translate(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """Translate a list of texts"""
        translations = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.processing_class(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate translations
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True
                )

            # Decode
            batch_translations = self.processing_class.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(batch_translations)

        return translations

    def evaluate_flores_dataset(self, eval_data) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Evaluate on Flores evaluation dataset"""

        # Extract English texts
        english_texts = eval_data["text_eng"].tolist()
        swahili_refs = eval_data["text_swh"].tolist()

        # Translate texts
        translations = self.translate(english_texts)

        # Compute metrics
        bleu_result = self.bleu_metric.compute(
            predictions=translations,
            references=[[ref] for ref in swahili_refs]
        )

        ter_result = self.ter_metric.compute(
            predictions=translations,
            references=[[ref] for ref in swahili_refs]
        )

        # Note on BLEU scores:
        # - SacreBLEU returns scores in the range 0-100, not 0-1
        # - This is standard practice and doesn't need normalization
        # - TER scores are also on their own scale (lower is better)
        results = {
            "bleu": round(bleu_result["score"], 4),  # Range is 0-100
            "ter": round(ter_result["score"], 4),   # Lower is better
        }

        # Create dataframe with examples
        examples_df = pd.DataFrame({
            "source": english_texts[:10],  # Show first 10 examples
            "reference": swahili_refs[:10],
            "translation": translations[:10]
        })

        return results, examples_df
