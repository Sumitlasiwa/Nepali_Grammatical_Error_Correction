"""
Dataset loading and preprocessing utilities
"""
from datasets import load_dataset
import torch
import random
import numpy as np

def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"✅ Seeds set to {seed}")


def load_and_prepare_dataset(config):
    """
    Load dataset and create train/validation splits
    
    Args:
        config: Config object with dataset settings
    
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    print(f"\n📚 Loading dataset: {config.dataset_name}")
    ds = load_dataset(config.dataset_name)
    
    # Select subset if specified
    if config.num_train_samples == None:
        ds["train"] = ds["train"].shuffle(seed=config.seed)
    else:
        ds["train"] = ds["train"].shuffle(seed=config.seed).select(range(config.num_train_samples))
    
    ds["validation"] = ds["validation"].shuffle(seed= config.seed).select(range(config.num_full_valid_samples)) # This is full validation samples acutal sub_val samples will be taken in trainer()   
    
    print(f"  Train: {len(ds['train'])} samples")
    print(f"  validation: {len(ds['validation'].select(range(config.num_valid_samples))) } samples")
    print(f"  Full validation: {len(ds['validation'])} samples")
    
    return ds


def preprocess_dataset(dataset, tokenizer, config):
    """
    Tokenize and prepare dataset for training
    
    Args:
        dataset: Dict with 'train' and 'validation' splits
        tokenizer: Tokenizer instance
        config: Config object
    
    Returns:
        Encoded dataset ready for training
    """
    print("\n⚙️  Preprocessing dataset...")
    
    def preprocess_batch(batch):
        # Add prefix to inputs
        inputs = [config.prefix + inp for inp in batch["incorrect_sentence"]]
        
        # Tokenize inputs
        input_encodings = tokenizer(
            inputs, 
            max_length=config.max_length,
            truncation=True 
        )
        
        # Tokenize targets
        
        target_encodings = tokenizer(
            batch["correct_sentence"], 
            max_length=config.max_length,
            truncation=True
        )
        
        # Set labels
        input_encodings["labels"] = target_encodings["input_ids"]
        return input_encodings
    
    # Process both splits
    encoded = {
        "train": dataset["train"].map(preprocess_batch, batched=True, batch_size=1000, num_proc=8),
        "validation": dataset["validation"].map(preprocess_batch, batched=True, batch_size=1000, num_proc=8)
    }
    
    # Set format for PyTorch
    for split in ["train", "validation"]:
        encoded[split].set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    print("  ✅ Preprocessing complete")
    return encoded

if __name__ == "__main__":
    from config import Config
    from trainer import setup_model
    config = Config()
    set_seeds()
    dataset = load_and_prepare_dataset(config)
    print(dataset['train'][0])
    _, tokenizer =setup_model(config)
    encoded = preprocess_dataset(dataset, tokenizer, config)
    print(encoded["train"][0])
    