                
import warnings
warnings.filterwarnings("ignore")

import torch
import os
import wandb
from math import ceil
from transformers import (
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# Import our custom modules
from config import Config
from metrics.metrics import create_compute_metrics
from data_utils import set_seeds, load_and_prepare_dataset, preprocess_dataset
from utils import clear_memory, create_directories, safe_training_check, save_model_safe
from trainer import setup_model, create_training_args
from callbacks import create_callbacks

def main():
    """Main training function"""
    
    # Load config
    config = Config()
    
    print("=" * 60)
    print("🚀 Nepali Grammar Error Correction Training")
    print("=" * 60)
    print(f"Model: {config.model_id}")
    print(f"LoRA: {config.use_lora}")
    print("=" * 60)
    
    # Setup
    set_seeds(config.seed)
    clear_memory()
    create_directories(config.output_dir)
    
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.project_name,
        config=vars(config),
    )
    run_id = wandb.run.id
    
    # Load data
    dataset = load_and_prepare_dataset(config)
    
    # Setup model
    model, tokenizer = setup_model(config)
    
    # Preprocess
    dataset_encoded = preprocess_dataset(dataset, tokenizer, config)
    
    # Create training args
    training_args = create_training_args(config, dataset_encoded, run_id)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,   #enables dynamic padding, pads to max_length in batch only
        pad_to_multiple_of=None,
        return_tensors="pt"
    )
    
    # Create metrics
    compute_metrics = create_compute_metrics(tokenizer, config, dataset["validation"])
    
    # Create callbacks
    callbacks = create_callbacks(config, tokenizer, dataset)
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["validation"].select(range(config.num_valid_samples)),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
        # Safety check
    if not safe_training_check(trainer):
        print("\n❌ Safety checks failed! Fix issues before training.")
        return
    
    # Train!
    print("\n" + "=" * 100)
    print("🏋️  Starting training...")
    print("=" * 100)
    
    try:
        if config.resume_from_checkpoint:
            print("continuing training from latest checkpoint.....")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        print("\n✅ Training complete!")
        # clear_memory()
     
        # print("Evaluating on full validation set...")
        # # Temporarily reduce batch size for evaluation
        # original_batch_size = trainer.args.per_device_eval_batch_size
        # trainer.args.per_device_eval_batch_size = 2  # Reduce batch size
        # trainer.compute_metrics = create_compute_metrics(tokenizer, config, dataset["validation"])
        
        
        # full_results = trainer.evaluate(eval_dataset=dataset_encoded["validation"],
        #                                 generation_max_length=config.max_length, #64
        #                                 generation_num_beams=config. #4
        #                                 eval_generation_num_beams,
        #                                 predict_with_generate=config.eval_predict_with_generate #True
        #                                 )
        
        # # Restore original batch size if needed
        # trainer.args.per_device_eval_batch_size = original_batch_size
        # print(f"Full validation results: {full_results}")
        # wandb.log({
        # f"full_eval/{k}": v for k, v in full_results.items()
        # })
        wandb.finish()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        wandb.finish()
        return
    
    # Save model
    best_model_path = f"{config.output_dir}/best_model"
    save_model_safe(model, tokenizer, best_model_path, use_lora=config.use_lora) 
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    print(f"\n🎉 All done! Model saved to {best_model_path}")
    

if __name__ == "__main__":
    # Required for Windows
    import multiprocessing
    multiprocessing.freeze_support()
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main()
