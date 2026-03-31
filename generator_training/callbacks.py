"""
Custom callbacks for training
"""
import torch
from transformers import TrainerCallback, EarlyStoppingCallback
# from transformers.integrations import ModelCheckpoint
import wandb
from config import Config

config = Config()

# best_model_checkpoint_callback = ModelCheckpoint(
#         dirpath=f"{config.output_dir}/checkpoints",  # Save checkpoints in the `checkpoints` directory
#         filename="best_model",  # Name for the checkpoint file
#         monitor="eval_loss",  # Metric to monitor (can be changed to other metrics like "accuracy" or "eval_bleu")
#         mode="min",  # Whether we want to minimize the metric (use "max" for metrics like accuracy)
#         save_best_only=True,  # Only save the best model (based on `monitor` metric)
#         save_weights_only=False,  # Save both model and tokenizer
#         save_top_k=1,  # Only keep the best checkpoint
#         verbose=1,  # Print information when saving the checkpoint
#     )

class SamplePredictionCallback(TrainerCallback):
    """Generate predictions on a few validation samples and log to W&B."""

    def __init__(self, tokenizer, eval_dataset, num_samples=5, max_length=64):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        self.max_length = max_length
        
    @torch.no_grad()
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation - generate sample predictions"""
        model = kwargs["model"]
        model.eval()
        device = model.device
        
        # Select sample rows
        samples = self.eval_dataset.select(range(min(self.num_samples, len(self.eval_dataset))))

        table = wandb.Table(columns=["Input", "Target", "Prediction", "Match"])

        for sample in samples:
            # Use the original raw text fields, not token IDs
            inp_text = sample["incorrect_sentence"]
            tgt_text = sample["correct_sentence"]
            
            # Tokenize individual sample
            tokenized = self.tokenizer(
                inp_text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=self.max_length
            ).to(device)
            
            # Safe generate (LoRA-friendly)
            with torch.cuda.amp.autocast(enabled=True):
                output_ids = model.generate(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    max_length=self.max_length,
                    num_beams=4
                )
                
            pred_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Check if prediction matches target
            match = "✅" if pred_text.strip() == tgt_text.strip() else "❌"

            table.add_data(inp_text, tgt_text, pred_text, match)

            # Cleanup
            del tokenized, output_ids
            torch.cuda.empty_cache()
            
        # Log to wandb
        wandb.log({"sample_predictions": table, "epoch": state.epoch})
        
        print(f"📊 Logged {self.num_samples} sample predictions to W&B")
        
class CSVLoggingCallback(TrainerCallback):
    """Log evaluation metrics to CSV file"""
    
    def __init__(self, output_file="eval_metrics.csv"):
        self.output_file = output_file
        self.first_eval = True
        
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation - log metrics to CSV"""
        import csv
        
        # Get evaluation metrics
        eval_metrics = kwargs.get("metrics", {})
        step = state.global_step
        epoch = state.epoch
        
        # Find the most recent train_loss from log_history
        train_loss = None
        for log_entry in reversed(state.log_history):
            if "loss" in log_entry:
                train_loss = log_entry["loss"]
                break
        
        # Prepare row data
        row = {
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_metrics.get("eval_loss"),
            "exact_match": eval_metrics.get("eval_Exact Match"),
            "P": eval_metrics.get("eval_precision"),
            "R": eval_metrics.get("eval_recall"),
            "F05": eval_metrics.get("eval_F05"),
            "gleu": eval_metrics.get("eval_gleu"),
            "bleu": eval_metrics.get("eval_bleu"),
        }
        
        # Write or append to CSV
        mode = 'w' if self.first_eval else 'a'
        
        with open(self.output_file, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            
            # Write header only on first evaluation of this run
            if self.first_eval:
                writer.writeheader()
                self.first_eval = False
                
            writer.writerow(row)
        
        action = "Created" if mode == 'w' else "Appended to"
        print(f"💾 {action} {self.output_file} at step {step}")              
class MemoryCleanupCallback(TrainerCallback):
    """Clean up GPU memory periodically"""
    
    def on_step_end(self, args, state, control, **kwargs):
        """Clean memory every N steps"""
        if state.global_step % 100 == 0:
            torch.cuda.empty_cache()
            
class LossOnlyEvalCallback(TrainerCallback):
    """
    Callback to trigger frequent loss-only evaluations between full metric evaluations
    """
    def __init__(self, loss_eval_steps, full_eval_steps):
        self.loss_eval_steps = loss_eval_steps
        self.full_eval_steps = full_eval_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        """Trigger evaluation at appropriate intervals"""
        current_step = state.global_step
        
        if current_step == 0:
            return control
            
        # Check what type of evaluation this should be
        is_full_eval_step = current_step % self.full_eval_steps == 0
        is_loss_eval_step = current_step % self.loss_eval_steps == 0
        
        if is_full_eval_step:
            # Full evaluation - let the normal eval happen
            control.should_evaluate = True
        elif is_loss_eval_step:
            # Loss-only evaluation - trigger it
            control.should_evaluate = True
            
        return control
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Modify evaluation settings based on step"""
        trainer = kwargs.get('trainer')
        current_step = state.global_step
        
        if trainer is not None:
            is_full_eval_step = current_step % self.full_eval_steps == 0
            
            if not is_full_eval_step:
                # Loss-only evaluation
                print(f"\n📊 Step {current_step}: Loss-only evaluation (no metrics)")
                # Store originals
                trainer._stored_compute_metrics = trainer.compute_metrics
                trainer._stored_predict_with_generate = args.predict_with_generate
                
                # Disable metrics temporarily
                trainer.compute_metrics = None
                args.predict_with_generate = False
            else:
                print(f"\n📈 Step {current_step}: Full evaluation with metrics")
                # Ensure metrics are enabled for full eval
                if hasattr(trainer, '_stored_compute_metrics'):
                    trainer.compute_metrics = trainer._stored_compute_metrics
                    args.predict_with_generate = trainer._stored_predict_with_generate
        
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Restore settings after logging"""
        trainer = kwargs.get('trainer')
        
        if trainer is not None and hasattr(trainer, '_stored_compute_metrics'):
            # Restore original settings after evaluation
            trainer.compute_metrics = trainer._stored_compute_metrics
            args.predict_with_generate = trainer._stored_predict_with_generate
            # Clean up stored attributes
            delattr(trainer, '_stored_compute_metrics')
            delattr(trainer, '_stored_predict_with_generate')
        
        return control
    
    
def create_callbacks(config, tokenizer, dataset):
    """
    Create training callbacks based on config
    
    Args:
        config: Config object
        tokenizer: Tokenizer instance
        dataset: Dictionary with 'train' and 'validation' datasets (raw, not encoded)
    
    Returns:
        List of callback instances
    """
    callbacks = []
    
    # Early stopping (always included)
    callbacks.append(
        EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
    )
    print(f"✅ Added EarlyStoppingCallback (patience={config.early_stopping_patience})")
    
    # Sample prediction callback
    if config.use_sample_prediction_callback:
        callbacks.append(
            SamplePredictionCallback(
                tokenizer=tokenizer,
                eval_dataset=dataset["validation"],  # Use raw dataset
                num_samples=config.sample_prediction_num_samples,
                max_length=config.max_length
            )
        )
        print(f"✅ Added SamplePredictionCallback (num_samples={config.sample_prediction_num_samples})")
        
    
      # Saves csv file with metrics on each evaluation
    if config.use_CSVLoggingCallback:
        callbacks.append(
            CSVLoggingCallback()
        )
        
    # Memory cleanup callback
    if config.use_memory_cleanup_callback:
        callbacks.append(MemoryCleanupCallback())
        print(f"✅ Added MemoryCleanupCallback")
        
    
    loss_eval_steps = config.loss_eval_steps
    full_eval_steps = config.eval_steps
    
    loss_only_callback = LossOnlyEvalCallback(
        loss_eval_steps=loss_eval_steps,
        full_eval_steps=full_eval_steps
    )
    callbacks.append(loss_only_callback)
    
    print(f"\n📋 Evaluation Strategy:")
    print(f"  - Loss-only evaluation every {loss_eval_steps} steps")
    print(f"  - Full evaluation (with metrics) every {full_eval_steps} steps")
        
    # callbacks.append(best_model_checkpoint_callback)

    return callbacks