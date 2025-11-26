# English-Swahili Machine Translation with SRL Augmentation

This project investigates whether semantic role labeling (SRL) augmentation improves English-Swahili machine translation quality using the MarianMT architecture. Results show significant improvements in translation quality, with the SRL-augmented model achieving a BLEU score of 28.85 compared to 27.05 for the baseline model (a 6.6% relative improvement).

## Project Structure

- `main.py` - Main script for training and evaluation
- `preprocessing/` - Data loading and SRL augmentation
- `model/` - MarianMT model training and evaluation components
- `cached_models/` - Local storage for pre-trained models to avoid rate limits

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `token.txt` file with your Hugging Face token in the project root.

## Data Management

The project handles three distinct datasets:

1. **Training Set**: The main dataset used to train the models, derived from the JW300 corpus (via NLLB dataset). The training set is split into:
   - **Training Portion**: Used for model parameter updates (80% by default)
   - **Validation Portion**: Used to evaluate the model during training (20% by default)

2. **FLORES Evaluation Set**: A completely separate dataset used only for final model evaluation. This ensures unbiased assessment of model performance on unseen data.

The `--validation-split` parameter controls the ratio of training/validation data. For example, with a value of 0.2, 80% of the data is used for training and 20% for validation.

When using `--max-train-samples`, the system automatically scales both training and validation sets proportionally to maintain the validation ratio.

### Notes on BLEU Score

The BLEU scores are reported on a 0-100 scale, which is the standard convention for machine translation evaluation, rather than a 0-1 scale.

## Training Options

### Parameter Explanation

- **Batch Size**: Number of samples processed at once. Larger values (16-32) speed up training but require more GPU memory. For an RTX 5080, a batch size of 16 works well.

- **Gradient Accumulation**: Number of forward passes before updating weights. This effectively multiplies batch size without increasing memory usage. A value of 4 means an effective batch size of 4 × batch_size.

- **Validation Split**: Fraction of training data (0-1) used for validation during training. Default is 0.2 (20%). The model trains on the remaining 80% and evaluates on the validation set during training to monitor progress.

- **Epochs**: Number of complete passes through the training dataset. More epochs generally improve results but increase training time. For the full dataset, 3 epochs is typically sufficient.

- **Freezing Encoder Layers**: Keeps encoder weights fixed, which:
  - Speeds up training significantly (2-3× faster)
  - Preserves the model's pre-trained understanding of English
  - Generally produces better results for low-resource languages

- **FP16 Precision**: Uses half-precision floating point, which:
  - Reduces memory usage by up to 50%
  - Accelerates training considerably on modern GPUs
  - Has minimal impact on translation quality

## Results

### Model Performance

| Model         | BLEU Score | TER Score | Relative Improvement |
|---------------|------------|-----------|----------------------|
| Baseline      | 27.05      | 60.15     | -                    |
| SRL-augmented | 28.85      | 56.76     | +6.6%                |

The results demonstrate that adding semantic role labeling information improves translation quality, with the effects becoming more pronounced with larger training datasets and more training epochs.

## Running the Model

### Full Training (Best Results)

To train on the complete dataset for the best possible results:

```bash
python main.py --full-train --batch-size 64 --grad-accum 2 --epochs 5 --fp16 --validation-split 0.1
```

This configuration:
- Uses all (augmented) training samples
- Employs an effective batch size of 128 (32 × 4)
- Runs for 3 complete epochs
- Sets aside 10% of data for validation
- Freezes encoder layers by default
- Uses FP16 precision for faster training
- May take 9-10 hours

### Medium Training (Good Balance)

For a good balance between training time and model quality:

```bash
python main.py --max-train-samples 50000 --batch-size 32 --grad-accum 4 --epochs 2 --fp16 --validation-split 0.2
```

This configuration:
- Uses approximately 50,000 total samples (40,000 for training, 10,000 for validation)
- Takes 1-2 hours to complete
- Still produces decent translation quality

### Quick Experiment (Fast Iteration)

For fast testing and experimentation:

```bash
python main.py --max-train-samples 5000 --batch-size 16 --grad-accum 2 --epochs 1 --fp16 --validation-split 0.2
```

This configuration:
- Uses approximately 5,000 total samples (4,000 for training, 1,000 for validation)
- Takes 10-15 minutes to complete
- Useful for debugging or testing changes

### Additional Options

- `--no-freeze-encoder`: Disable encoder layer freezing (not recommended)
- `--disable-tqdm`: Suppress progress bars for cleaner output logs
- `--max-steps N`: Limit training to N steps (overrides epochs)
- `--cpu`: Force CPU training (very slow, not recommended)
- `--validation-split X`: Set the fraction of data used for validation (default: 0.2)

## Evaluation

Evaluation runs automatically after training. The process:

1. Evaluates both baseline and SRL-augmented models
2. Calculates BLEU and TER metrics on the FLORES test set
3. Provides side-by-side translation examples for comparison
4. Saves results to `translation_comparison.txt`

A higher BLEU score and lower TER score indicate better translation quality.

## Testing

To verify your setup with a minimal test:

```bash
python -m model.test
```

This runs a small-scale training and evaluation to ensure everything is configured correctly.

## Caching

The project implements model caching to avoid Hugging Face rate limits. Pre-trained models are downloaded once and stored in the `cached_models` directory for future use.
