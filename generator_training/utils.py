"""
Utility functions for training
"""
import os
import torch
import gc

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared")
    
def create_directories(output_dir):
    """Create necessary output directories"""
    dirs = [
        os.path.join(output_dir, "checkpoints"),
        os.path.join(output_dir, "best_model"),
        os.path.join(output_dir, "logs"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Directories created in {output_dir}")
    
def safe_training_check(trainer):
    """
    Comprehensive pre-training safety check
    
    Usage:
        safe_training_check(trainer)
    Returns:
        bool: True if all checks pass
    """
    print("\n Running pre-training safety checks...")

    # 1. Check model is on correct device
    print(f"Model device: {next(trainer.model.parameters()).device}")

    # 2. Check dataset sizes
    print(f"Train dataset size: {len(trainer.train_dataset)}")
    print(f"Eval dataset size: {len(trainer.eval_dataset)}")

    # 3. Test data loading
    try:
        sample_batch = next(iter(trainer.get_train_dataloader()))
        print(" Data loading works")
       
    except Exception as e:
        print(f" Data loading failed: {e}")
        return False

    # 4. Test evaluation
    try:
        trainer.model.eval()    # Set to evaluation mode
        print(" Performing evaluation check...")
        eval_results = trainer.evaluate()
        print(" Evaluation successful")
        print(f"Initial metrics: {eval_results}")
        return True
    except Exception as e:
        print(f" Evaluation failed: {e}")
        return False

    
def save_model_safe(model, tokenizer, best_model_path, use_lora):
    """
    Safely save model handling LoRA and gradient checkpointing
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Where to save
        use_lora: Whether model uses LoRA
    """
    print(f"\n💾 Saving model to {best_model_path}...")
    os.makedirs(best_model_path, exist_ok=True)
    
    try:
        if use_lora:
            # LoRA: save adapter only
            model.save_pretrained(best_model_path)
            print("  ✅ LoRA adapter saved")
        else:
            # Full model: disable gradient checkpointing first
            if hasattr(model, 'gradient_checkpointing_disable'):
                model.gradient_checkpointing_disable()
                model.save_pretrained(best_model_path, safe_serialization=True)
            print("  ✅ Full model saved")
        
        # Save tokenizer
        tokenizer.save_pretrained(best_model_path)
        print("  ✅ Tokenizer saved")
        
    except Exception as e:
        print(f"  ⚠️  Save failed: {e}")
        print("  Trying fallback method...")
        torch.save(model.state_dict(), os.path.join(best_model_path, "model_state.pt"))
        tokenizer.save_pretrained(best_model_path)
        print("  ✅ State dict saved as fallback")
        
