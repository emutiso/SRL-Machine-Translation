import os
import pandas as pd
import numpy as np
from huggingface_hub import login
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from preprocessing.srl_augmenter import batch_augment, device
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read the token from the text file
token_path = PROJECT_ROOT / 'token.txt'
with open(token_path, 'r') as file:
    token = file.read().strip()

# Needed for dataset access
login(token=token)

def augment_srl(dataset):
    """Batch processing for SRL augmentation"""

    # Filter incase data is missing
    dataset = dataset.filter(
        lambda x: x["translation"]["eng_Latn"] is not None,
        num_proc=4
    )

    # Process a batch of examples
    def process_batch(batch):
        eng_texts = [item["eng_Latn"] for item in batch["translation"]]
        swh_texts = [item["swh_Latn"] for item in batch["translation"]]

        # Augment with SRL tags
        augmented_texts = batch_augment(eng_texts)

        return {
            "translation": [
                {"eng_Latn": a, "swh_Latn": s}
                for a, s in zip(augmented_texts, swh_texts)
            ]
        }

    # Use full dataset or limit sample size
    samples = 1200000  # Increased to 1.2 million
    if len(dataset) > samples:
        dataset = dataset.select(range(samples))

    # Batch size = 64, num_proc = 6 Worked best SO FAR
    return dataset.map(
        process_batch,
        batched=True,
        batch_size=64,
        num_proc=6 if device.type == "cuda" else 4
    )

def preprocess_and_save_parquet():
    """Preprocess and save evaluation data to Parquet files"""

    # Load the flores_plus dataset with default config
    dataset = load_dataset("openlanguagedata/flores_plus", "default", split="devtest")
    print("Loaded flores_plus evaluation data.")

    # Filter the dataset for English and Swahili texts
    eng_dataset = dataset.filter(lambda example: example['iso_639_3'] == 'eng')
    swh_dataset = dataset.filter(lambda example: example['iso_639_3'] == 'swh')

    # Convert to pandas DataFrame for easier manipulation
    eng_df = pd.DataFrame(eng_dataset)
    swh_df = pd.DataFrame(swh_dataset)

    # Merge the datasets on the 'id' column
    merged_df = pd.merge(eng_df[['id', 'text']], swh_df[['id', 'text']], on='id', suffixes=('_eng', '_swh'))

    # Save the merged DataFrame to a Parquet file
    merged_df.to_parquet(PROJECT_ROOT / 'datasets' / 'flores_plus_eval_no_srl.parquet')
    print("Saved preprocessed non-augmented dataset to flores_plus_eval_no_srl.parquet.")

    # Create and save SRL-augmented version
    print("Augmenting evaluation data...")
    eng_texts = merged_df['text_eng'].tolist()
    merged_df_srl = merged_df.copy()
    merged_df_srl['text_eng'] = batch_augment(eng_texts)
    merged_df_srl.to_parquet(PROJECT_ROOT / 'datasets' / 'flores_plus_eval_srl.parquet')
    print("Saved preprocessed augmented dataset to flores_plus_eval_no_srl.parquet.")

    print("\nSaved both evaluation datasets.\n")

def load_eval_data(srl_augment=True):
    """Load preprocessed Flores+ evaluation data"""
    parquet_name = "flores_plus_eval_srl.parquet" if srl_augment else "flores_plus_eval_no_srl.parquet"
    parquet_path = PROJECT_ROOT / 'datasets' / parquet_name

    if not os.path.exists(parquet_path):
        preprocess_and_save_parquet()

    evals = pd.read_parquet(parquet_path)
    print(f"Loaded {'SRL-augmented' if srl_augment else 'non-augmented'} evaluation data.")
    return evals

def load_train_data(srl_augment=True):
    """Load NLLB data with/without SRL augmentation (Same as JW300 training data)"""
    dataset_path = PROJECT_ROOT / 'datasets' / ('nllb_train_srl' if srl_augment else 'nllb_train_no_srl')

    try:
        # Try to load the dataset from the local path
        train = Dataset.load_from_disk(dataset_path)
        print(f"Loaded {'augmented' if srl_augment else 'non-augmented'} train data from disk.")
    except FileNotFoundError:
        # If not found, download and save it
        train = load_dataset(
            "allenai/nllb",
            "eng_Latn-swh_Latn",
            split="train",
            verification_mode="all_checks",
            trust_remote_code=True  # Allow custom code to be run
        )

        # Use full dataset or limit sample size
        samples = 1200000  # Increased to 1.2 million
        if len(train) > samples:
            train = train.select(range(samples))

        # Remove unnecessary columns
        train = train.remove_columns([col for col in train.column_names if col != 'translation'])

        if srl_augment:
            print("Augmenting training data with SRL tags...")
            train = augment_srl(train)

        train.save_to_disk(dataset_path)
        print(f"Downloaded and saved {'augmented' if srl_augment else 'non-augmented'} train data to disk.")


    return train

def split_dataset(dataset, val_size=0.1, seed=42):
    """Split a dataset into training and validation sets
    
    Args:
        dataset: The dataset to split
        val_size: The proportion of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset: The split datasets
    """
    # Get dataset length
    dataset_size = len(dataset)
    
    # Create indices for train/val split
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, random_state=seed
    )
    
    # Split dataset using indices
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    
    print(f"Split dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    return train_dataset, val_dataset

def get_data(val_split=0.1):
    """Get training, validation, and evaluation data
    
    Args:
        val_split: Proportion of training data to use for validation
        
    Returns:
        Tuple containing:
        - train_data_no_srl: Non-augmented training data
        - val_data_no_srl: Non-augmented validation data
        - train_data_srl: SRL-augmented training data
        - val_data_srl: SRL-augmented validation data
        - flores_data_no_srl: Non-augmented FLORES evaluation data
        - flores_data_srl: SRL-augmented FLORES evaluation data
    """
    # Load full training datasets
    full_train_no_srl = load_train_data(srl_augment=False)
    full_train_srl = load_train_data(srl_augment=True)
    
    # Split into train/validation sets
    train_data_no_srl, val_data_no_srl = split_dataset(full_train_no_srl, val_size=val_split)
    train_data_srl, val_data_srl = split_dataset(full_train_srl, val_size=val_split)
    
    # Load FLORES evaluation data (separate test set)
    flores_data_no_srl = load_eval_data(srl_augment=False)
    flores_data_srl = load_eval_data(srl_augment=True)
    
    return (
        train_data_no_srl, 
        val_data_no_srl, 
        train_data_srl, 
        val_data_srl,
        flores_data_no_srl, 
        flores_data_srl
    )

# Sample Outputs
if __name__ == "__main__":
    train_data_no_srl, val_data_no_srl, train_data_srl, val_data_srl, flores_data_no_srl, flores_data_srl = get_data()

    print(f"\nTraining samples (Non-augmented): {len(train_data_no_srl)}")
    print(f"Validation samples (Non-augmented): {len(val_data_no_srl)}")
    print(f"Training samples (Augmented): {len(train_data_srl)}")
    print(f"Validation samples (Augmented): {len(val_data_srl)}")
    print(f"FLORES evaluation samples (Non-augmented): {len(flores_data_no_srl)}")
    print(f"FLORES evaluation samples (Augmented): {len(flores_data_srl)}")

    # Show sample evaluation entry
    print("\nSample FLORES evaluation sentence:")
    print(f"EN (Non-Augmented): {flores_data_no_srl.iloc[0]['text_eng']}")
    print(f"EN (Augmented): {flores_data_srl.iloc[0]['text_eng']}")
    print(f"SW: {flores_data_srl.iloc[0]['text_swh']}")

    # Show sample training entry
    print("\nSample training sentence:")
    print(train_data_no_srl[0])
    print(train_data_srl[0])

    print("\nSample validation sentence:")
    print(val_data_no_srl[0])
    print(val_data_srl[0])
