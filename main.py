# Combines steps for final result
from preprocessing.data_loader import get_data
from model.model_trainer import MarianMTTrainer
from model.model_evaluator import MarianMTEvaluator
from model.model_utils import set_seed, compare_translations
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = os.path.join(PROJECT_ROOT, "cached_models")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MarianMT fine-tuning for English-Swahili translation with SRL augmentation")
    
    # Dataset size options
    parser.add_argument("--full-train", action="store_true", help="Run on the full training dataset (300k samples)")
    parser.add_argument("--max-train-samples", type=int, default=5000, help="Maximum number of training samples")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Fraction of training data to use for validation (0-1)")
    
    # Training configuration options
    parser.add_argument("--batch-size", type=int, default=48, help="Batch size for processing")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (multiplies effective batch size)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--no-freeze-encoder", action="store_true", help="Disable encoder layer freezing (freezing is ON by default)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Maximum number of training steps (overrides epochs if set)")
    
    # Hardware/optimization options
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision for faster training")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (not recommended)")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable progress bars for cleaner output")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR, help="Directory to cache models")
    
    # Skip options for faster iterations
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline model training and use previously trained model")
    parser.add_argument("--baseline-model-path", type=str, default="models/marianmt_baseline", 
                        help="Path to pre-trained baseline model (used with --skip-baseline)")
    
    return parser.parse_args()

def main():
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Ensure cache directory exists
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"Using model cache directory: {args.cache_dir}")
        print(f"Run timestamp: {timestamp}")
    except Exception as e:
        print(f"Error: {e}")

    # Get training, validation, and evaluation data
    print("Loading data...")
    train_data_no_srl, val_data_no_srl, train_data_srl, val_data_srl, flores_data_no_srl, flores_data_srl = get_data(val_split=args.validation_split)

    # Print confirmation messages
    print(f"\nTraining samples (Non-augmented): {len(train_data_no_srl)}")
    print(f"Validation samples (Non-augmented): {len(val_data_no_srl)}")
    print(f"Training samples (Augmented): {len(train_data_srl)}")
    print(f"Validation samples (Augmented): {len(val_data_srl)}")
    print(f"FLORES evaluation samples (Non-augmented): {len(flores_data_no_srl.index)}")
    print(f"FLORES evaluation samples (Augmented): {len(flores_data_srl.index)}")

    # Define compute_metrics function
    def compute_metrics(eval_preds):
        evaluator = MarianMTEvaluator(
            model_path="Helsinki-NLP/opus-mt-en-sw",
            use_cuda=not args.cpu,
            cache_dir=args.cache_dir
        )
        return evaluator.compute_metrics(eval_preds)

    # Train baseline model (unless --skip-baseline is specified)
    if args.skip_baseline:
        print("\n\n" + "=" * 50)
        print("Skipping baseline model training (--skip-baseline flag set)")
        print("=" * 50)
        
        # Check if baseline model exists
        baseline_model_path = args.baseline_model_path
        if not os.path.exists(baseline_model_path):
            print(f"ERROR: Baseline model not found at {baseline_model_path}")
            print("Please train a baseline model first or specify a correct path with --baseline-model-path")
            return
            
        print(f"Using pre-trained baseline model from: {baseline_model_path}")
    else:
        print("\n\n" + "=" * 50)
        print("Training baseline model (without SRL)")
        print("=" * 50)

    # Determine train sizes based on full-train flag
    max_train_samples = None if args.full_train else args.max_train_samples
    
    # When using max_train_samples, we need to limit both train and validation datasets
    if max_train_samples is not None:
        total_samples_before = len(train_data_no_srl) + len(val_data_no_srl)
        ratio = max_train_samples / total_samples_before
        
        # Resize each dataset by the same ratio to maintain validation split
        new_train_size = int(len(train_data_no_srl) * ratio)
        new_val_size = int(len(val_data_no_srl) * ratio)
        
        print(f"Limiting training data from {len(train_data_no_srl)} to {new_train_size} samples")
        print(f"Limiting validation data from {len(val_data_no_srl)} to {new_val_size} samples")
        
        train_data_no_srl = train_data_no_srl.select(range(new_train_size))
        val_data_no_srl = val_data_no_srl.select(range(new_val_size))
        train_data_srl = train_data_srl.select(range(new_train_size))
        val_data_srl = val_data_srl.select(range(new_val_size))
    
    # Only train the baseline model if not skipped
    if not args.skip_baseline:
        # For baseline model, keep consistent path without timestamp
        baseline_output_dir = os.path.join("models", "marianmt_baseline")
        os.makedirs(baseline_output_dir, exist_ok=True)
        
        baseline_trainer = MarianMTTrainer(
            use_srl=False,
            output_dir=baseline_output_dir,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            epochs=args.epochs,
            max_steps=args.max_steps,
            freeze_encoder=not args.no_freeze_encoder,  # Inverted logic: we freeze by default unless --no-freeze-encoder is specified
            fp16=args.fp16 and not args.cpu,
            use_cuda=not args.cpu,
            disable_tqdm=args.disable_tqdm,
            cache_dir=args.cache_dir
        )

        try:
            baseline_trainer, baseline_model_path = baseline_trainer.train(
                train_dataset=train_data_no_srl,
                eval_dataset=val_data_no_srl,  # Use validation data for evaluation during training
                compute_metrics=compute_metrics
            )
        except Exception as e:
            print(f"ERROR during baseline model training: {str(e)}")
            print("Please check logs for details or try with a smaller dataset.")
            return
    else:
        # Use the specified path if skipping training
        baseline_model_path = args.baseline_model_path

    # Train SRL-augmented model
    print("\n\n" + "=" * 50)
    print("Training SRL-augmented model")
    print("=" * 50)
    
    # Create timestamped output directory
    timestamped_output_dir = os.path.join("models", f"marianmt_srl_{timestamp}")
    os.makedirs(timestamped_output_dir, exist_ok=True)

    srl_trainer = MarianMTTrainer(
        use_srl=True,  # SRL is always used with special tokens now
        output_dir=timestamped_output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        epochs=args.epochs,
        max_steps=args.max_steps,
        freeze_encoder=not args.no_freeze_encoder,  # Inverted logic: we freeze by default unless --no-freeze-encoder is specified
        fp16=args.fp16 and not args.cpu,
        use_cuda=not args.cpu,
        disable_tqdm=args.disable_tqdm,
        cache_dir=args.cache_dir
    )

    try:
        srl_trainer, srl_model_path = srl_trainer.train(
            train_dataset=train_data_srl,
            eval_dataset=val_data_srl,  # Use validation data for evaluation during training
            compute_metrics=compute_metrics
        )
    except Exception as e:
        print(f"ERROR during SRL-augmented model training: {str(e)}")
        print("Please check logs for details or try with a smaller dataset.")
        return

    # Evaluate both models on FLORES test set
    print("\n\n" + "=" * 50)
    print("Evaluating models on FLORES test set")
    print("=" * 50)

    # Evaluate baseline model
    baseline_evaluator = MarianMTEvaluator(
        model_path=baseline_model_path,
        use_cuda=not args.cpu,
        cache_dir=args.cache_dir
    )

    # Use a smaller subset of the FLORES evaluation data for speed
    eval_subset_size = len(flores_data_no_srl)
    flores_data_no_srl_subset = flores_data_no_srl.head(eval_subset_size)
    flores_data_srl_subset = flores_data_srl.head(eval_subset_size)

    baseline_results, baseline_examples = baseline_evaluator.evaluate_flores_dataset(flores_data_no_srl_subset)

    # Evaluate SRL-augmented model
    srl_evaluator = MarianMTEvaluator(
        model_path=srl_model_path,
        use_cuda=not args.cpu,
        cache_dir=args.cache_dir
    )

    srl_results, srl_examples = srl_evaluator.evaluate_flores_dataset(flores_data_srl_subset)

    # Print evaluation results
    print("\nBaseline Model Results:")
    for metric, value in baseline_results.items():
        print(f"- {metric}: {value}")

    print("\nSRL-Augmented Model Results:")
    for metric, value in srl_results.items():
        print(f"- {metric}: {value}")

    # Compare translations
    print("\nTranslation Examples:")
    sample_sources = flores_data_no_srl_subset["text_eng"].tolist()[:5]
    sample_refs = flores_data_no_srl_subset["text_swh"].tolist()[:5]
    baseline_translations = baseline_evaluator.translate(sample_sources)
    
    # For SRL model, use SRL-augmented source texts
    srl_sample_sources = flores_data_srl_subset["text_eng"].tolist()[:5]
    srl_translations = srl_evaluator.translate(srl_sample_sources)

    comparison = compare_translations(
        source_texts=sample_sources,
        reference_texts=sample_refs,
        baseline_translations=baseline_translations,
        srl_translations=srl_translations,
        n_examples=5
    )

    print(comparison)

    # Save comparison to file with timestamp
    comparison_file = f"translation_comparison_{timestamp}.txt"
    with open(comparison_file, "w", encoding="utf-8") as f:
        f.write(comparison)
        
    # Save evaluation metrics to a separate file with timestamp
    metrics_file = f"evaluation_metrics_{timestamp}.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("# English-Swahili Machine Translation Evaluation Results\n\n")
        f.write("## Baseline Model (No SRL)\n")
        f.write(f"- BLEU Score: {baseline_results['bleu']}\n")
        f.write(f"- TER Score: {baseline_results['ter']}\n\n")
        f.write("## SRL-Augmented Model\n")
        f.write(f"- BLEU Score: {srl_results['bleu']}\n")
        f.write(f"- TER Score: {srl_results['ter']}\n")
    
    # Generate 50 random samples for human evaluation
    import random
    
    # Select 50 random indices (or all if fewer than 50 samples)
    max_samples = min(50, len(flores_data_no_srl))
    random_indices = random.sample(range(len(flores_data_no_srl)), max_samples)
    
    # Translate the selected samples with both models
    random_sources = [flores_data_no_srl.iloc[idx]['text_eng'] for idx in random_indices]
    random_refs = [flores_data_no_srl.iloc[idx]['text_swh'] for idx in random_indices]
    random_srl_sources = [flores_data_srl.iloc[idx]['text_eng'] for idx in random_indices]
    
    random_baseline_translations = baseline_evaluator.translate(random_sources)
    random_srl_translations = srl_evaluator.translate(random_srl_sources)
    
    # Save the samples to a file for human evaluation with timestamp
    human_eval_file = f"human_evaluation_samples_{timestamp}.txt"
    with open(human_eval_file, "w", encoding="utf-8") as f:
        f.write("# Human Evaluation Samples - 50 Random Sentences\n\n")
        
        for i, idx in enumerate(random_indices, 1):
            src = flores_data_no_srl.iloc[idx]['text_eng']
            ref = flores_data_no_srl.iloc[idx]['text_swh']
            src_srl = flores_data_srl.iloc[idx]['text_eng']
            baseline_trans = random_baseline_translations[i-1]
            srl_trans = random_srl_translations[i-1]
            
            f.write(f"## Sample {i}\n")
            f.write(f"Source: {src}\n")
            f.write(f"SRL-Tagged: {src_srl}\n")
            f.write(f"Reference: {ref}\n\n")
            f.write(f"Baseline Translation: {baseline_trans}\n")
            f.write(f"SRL-Augmented Translation: {srl_trans}\n")
            f.write("Human Rating (1-5): [ ]\n")
            f.write("Comments: \n\n")
    
    print("\nEvaluation complete. Results saved to:")
    print(f"1. {comparison_file} - Sample translations for comparison")
    print(f"2. {metrics_file} - BLEU and TER scores for both models")
    print(f"3. {human_eval_file} - 50 random samples for human evaluation")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
