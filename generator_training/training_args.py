"""
Main training script for Nepali GEC
Keep this file clean - all logic is in other modules!
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import wandb
from math import ceil
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments
    )
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel


# Import our custom modules
from config import Config
from metrics.metrics import create_compute_metrics
from data_utils import set_seeds, load_and_prepare_dataset, preprocess_dataset
from utils import clear_memory, create_directories, safe_training_check, save_model_safe


def setup_model(config):
    """Load and configure model based on settings"""
    print(f"\n Loading model: {config.model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=False, legacy=False)
    
    if config.use_lora:
        if config.load_in_8bit:
            print("Using LoRA + 8-bit model loading !")
        elif config.load_in_4bit:
            print("Using QLoRA !")
        else:
            print(" Using LoRA only !")
        
        dtype_arg = torch.bfloat16 if (not config.load_in_8bit and not config.load_in_4bit) else None

        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_id,
            dtype=dtype_arg,
            # load_in_8bit = config.load_in_8bit,
            # load_in_4bit = config.load_in_4bit,
            device_map="auto"
        ) 
        
        # Prepare for LoRA
        model = prepare_model_for_kbit_training(model)
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Add LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        model.print_trainable_parameters()
        

        # adapter_path = r"C:\Users\Lenovo\Desktop\Nepali_GEC\nepali_gec\outputs\best_model_A_v2l"  # folder containing adapter_model.bin
        # model = PeftModel.from_pretrained(
        #     model,             # base model + LoRA config
        #     adapter_path,      # LoRA checkpoint
        #     is_trainable=True  # continue training
        # )

    else:
        print("  Using full fine-tuning")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_id,
            torch_dtype=None, #torch.float16 if config.use_fp16 else 
            device_map=None,
        )
        model.to("cuda")
    
    return model, tokenizer

def create_training_args(config, dataset_encoded, run_id):
    """Create training arguments from config"""
    
    # Calculate steps
    steps_per_epoch = ceil(len(dataset_encoded["train"]) / 
                          (config.batch_size * config.gradient_accumulation_steps))
    num_training_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(config.warmup_ratio * num_training_steps)
    
    print(f"\n📊 Training plan:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {num_training_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    return Seq2SeqTrainingArguments(
        output_dir=f"{config.output_dir}/checkpoints",
        num_train_epochs=config.num_epochs,
        
        # Batch & optimization
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        label_smoothing_factor =config.label_smoothing_factor,
        
        # Memory & speed
        fp16=config.use_fp16,
        # fp16_opt_level="01",
        # half_precision_backend="auto",
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_num_workers=config.dataloader_num_workers,
        optim="paged_adamw_8bit" if config.use_lora else "adamw_torch",
        
        # Logging & saving
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_only_model=True,
        
        # Best model
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        
        # Generation
        predict_with_generate=config.train_predict_with_generate,
        generation_max_length=config.max_length,
        generation_num_beams=config.train_generation_num_beams,
        
        # Reproducibility
        seed=config.seed,
        data_seed=config.seed,
        
        # Logging
        report_to="wandb",
        run_name=run_id,
        push_to_hub=False,
        # overwrite_output_dir=True,
    )
    


