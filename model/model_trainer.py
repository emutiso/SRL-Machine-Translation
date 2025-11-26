"""
Model trainer module for fine-tuning MarianMT models
"""

import os
import shutil
import torch
from pathlib import Path
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from typing import Dict, List, Optional, Union, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = os.path.join(PROJECT_ROOT, "cached_models")


class MarianMTTrainer:
    def __init__(
            self,
            model_name: str = "Helsinki-NLP/opus-mt-en-sw",
            output_dir: str = "models",
            use_srl: bool = False,
            max_length: int = 128,
            batch_size: int = 16,  # Default batch size for optimal performance
            gradient_accumulation_steps: int = 4,  # Effective batch size multiplier
            learning_rate: float = 2e-5,
            weight_decay: float = 0.01,
            epochs: int = 3,
            save_strategy: str = "epoch",
            use_cuda: bool = True,
            fp16: bool = True,
            freeze_encoder: bool = True,  # Freezing encoder by default for faster training
            max_steps: Optional[int] = -1,  # Limit number of training steps
            disable_tqdm: bool = False,  # Option to disable progress bars
            cache_dir: str = CACHE_DIR  # Directory to cache models
    ):
        self.model_name = model_name
        self.output_dir = os.path.join(PROJECT_ROOT, output_dir)
        self.use_srl = use_srl
        self.max_length = max_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.save_strategy = save_strategy
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.fp16 = fp16 and self.device.type == "cuda"
        self.freeze_encoder = freeze_encoder
        self.max_steps = max_steps
        self.disable_tqdm = disable_tqdm
        self.cache_dir = cache_dir

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load tokenizer and model (with caching)
        self.tokenizer, self.model = self._load_model_and_tokenizer()
        
        # Add SRL special tokens if using SRL data
        # This is always done when use_srl=True, as it's essential for the project
        if self.use_srl:
            self._add_srl_special_tokens()
            
            # Resize token embeddings to accommodate special tokens
            print("Resizing model embeddings to accommodate special tokens...")
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Optionally freeze encoder layers to speed up training
        if self.freeze_encoder:
            self._freeze_encoder_layers()
            
        self.model = self.model.to(self.device)

        # Set up data collator
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding="longest"
        )
        
        # Log configuration
        print(f"Model trainer initialized:")
        print(f" - Device: {self.device}")
        print(f" - Batch size: {self.batch_size}")
        print(f" - Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f" - Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        print(f" - FP16: {self.fp16}")
        print(f" - Encoder frozen: {self.freeze_encoder}")
        print(f" - Using SRL: {self.use_srl}")
        print(f" - Epochs: {self.epochs}")
        
    def _get_cache_path(self):
        """Get path for cached model"""
        # Replace slashes with underscores to create a valid filename
        model_dir_name = self.model_name.replace("/", "_")
        return os.path.join(self.cache_dir, model_dir_name)
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with caching to avoid repeated downloads"""
        cache_path = self._get_cache_path()
        
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
        print(f"Downloading model from Hugging Face: {self.model_name}")
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        model = MarianMTModel.from_pretrained(self.model_name)
        
        # Save to cache for future use
        print(f"Saving model to local cache: {cache_path}")
        tokenizer.save_pretrained(cache_path)
        model.save_pretrained(cache_path)
        
        return tokenizer, model

    def _add_srl_special_tokens(self):
        """Add SRL tags as special tokens to the tokenizer"""
        print("Adding SRL tags as special tokens...")
        
        # Define SRL tags as special tokens - fundamental to the project
        special_tokens = {
            "additional_special_tokens": [
                f"[Agent{i}]" for i in range(1, 7)
            ] + [
                f"[Verb{i}]" for i in range(1, 7)
            ] + [
                f"[Theme{i}]" for i in range(1, 7)
            ] + [
                f"[Goal{i}]" for i in range(1, 7)
            ]
        }
        
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens to the tokenizer vocabulary")
        
        # Show some of the added tokens for verification
        print(f"Added tokens include: {special_tokens['additional_special_tokens'][:5]}...")

    def _freeze_encoder_layers(self):
        """Freeze encoder layers to reduce training time"""
        print("Freezing encoder layers...")
        for param in self.model.get_encoder().parameters():
            param.requires_grad = False
        
    # Deprecated - now handled in main.py
    def limit_training_data(self, dataset, max_samples: int = None):
        """Optionally limit dataset size for faster training
        
        Note: This functionality is now handled in main.py with proper train/val splitting
        """
        # For backward compatibility
        return dataset

    def preprocess_data(self, dataset, num_proc: int = 4):
        """Preprocess the dataset for training"""

        def preprocess_function(examples):
            # For JW300/NLLB dataset format
            inputs = [ex["eng_Latn"] for ex in examples["translation"]]
            targets = [ex["swh_Latn"] for ex in examples["translation"]]

            # Tokenize inputs and targets using text_target parameter
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
            
            labels = self.tokenizer(
                text_target=targets,
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Apply preprocessing to the dataset
        return dataset.map(
            preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=["translation"]
        )

    def train(self, train_dataset, eval_dataset, 
              compute_metrics=None) -> Tuple:
        """Fine-tune the MarianMT model
        
        Args:
            train_dataset: Dataset for training
            eval_dataset: Dataset for evaluation (validation set)
            compute_metrics: Function to compute evaluation metrics
            
        Returns:
            Tuple of (trainer, output_dir)
        """
        # Dataset limiting is now handled in main.py

        # Use the provided output directory directly
        # This allows the main.py to set a timestamped directory
        output_dir = self.output_dir
        
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set up training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy=self.save_strategy,  # Fixed: using eval_strategy
            save_strategy=self.save_strategy,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            weight_decay=self.weight_decay,
            save_total_limit=3,
            num_train_epochs=self.epochs,
            max_steps=self.max_steps,  # Limit number of training steps
            predict_with_generate=True,
            fp16=self.fp16,
            load_best_model_at_end=True,
            metric_for_best_model="bleu" if compute_metrics else None,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            report_to="none",  # Disable wandb, etc.
            generation_config=None,  # Fix for generation config warning
            generation_max_length=self.max_length,  # Set generation parameters properly
            disable_tqdm=self.disable_tqdm  # Control progress bar display
        )

        # Preprocess datasets
        print(f"Preprocessing train dataset ({len(train_dataset)} samples)...")
        processed_train_dataset = self.preprocess_data(train_dataset)
        
        print(f"Preprocessing eval dataset ({len(eval_dataset)} samples)...")
        processed_eval_dataset = self.preprocess_data(eval_dataset)

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_train_dataset,
            eval_dataset=processed_eval_dataset,
            processing_class=self.tokenizer,  # Using processing_class instead of tokenizer
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )

        # Start training
        print(f"Starting training with {len(processed_train_dataset)} samples...")
        trainer.train()

        # Save the final model
        print(f"Training complete. Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return trainer, output_dir
