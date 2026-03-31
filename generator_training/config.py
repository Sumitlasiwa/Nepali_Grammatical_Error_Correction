from typing import List
from dataclasses import dataclass
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
# PROJECT_ROOT = "content" 

@dataclass
class Config:
    """Single configuration class for everything"""
    
    # Model settings
    model_id: str = "AIsumit123/mt5_base_v2_101"
    max_length: int = 64
    prefix: str = "Correct sentence: "
    
    # Dataset settings
    dataset_name: str = "AIsumit123/nepali_gec_data_v4"
    num_train_samples: int = None   # set to None for full train dataset
    num_valid_samples: int = 680 # quick valid set Cannot be set to none. Is subset of full valid set
    num_full_valid_samples: int = 680
    
    # LoRA settings
    use_lora: bool = False
    lora_r: int = 8         # small for small models
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Set to None for all layers
    
    # Training settings
    batch_size: int = 8
    num_epochs: int  = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    train_generation_num_beams: int = 1
    train_predict_with_generate: bool=True
    max_grad_norm: float = 0.5
    label_smoothing_factor: float = 0.0
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 0
    lr_scheduler_type: str = "cosine"
    metric_for_best_model: str="eval_loss"
    greater_is_better: bool=False
    loss_eval_steps: int = 20
    
    
    # Optimization settings
    load_in_8bit: bool = False   # 8 bit load + lora
    load_in_4bit: bool = False  # Qlora
    use_fp16: bool = False
    gradient_checkpointing: bool = False
    
    # Logging & saving
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs")
    metric_for_best_model: str ="eval_loss"
    greater_is_better: bool = False
    eval_steps: int = 50
    logging_steps: int = 20
    save_steps: int = 50
    save_total_limit: int = 4
    early_stopping_patience: int = 3
    
    # Wandb settings
    wandb_project: str = "nepali-grammar-correction"
    project_name: str = "mt5-base-v4_FF"
    
    # Seeds
    seed: int = 42
    
    def __post_init__(self):
        """Set default values that depend on other attributes"""
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q", "k","v", "o"] #, "v", "o", "wi_0", "wi_1", "wo"
            
    # Metrics configuratin -control which metric to compute
    bleu: bool = True
    gleu: bool = True
    Exact_Match: bool = True
    F05: bool = True
    
    def get_enabled_metrics(self) -> List[str]:
        """Get list of enabled metrics"""
        enabled = []
        
        if self.bleu:
            enabled.append("bleu")
        if self.F05:
            enabled.append("F05")
        if self.gleu:
            enabled.append("gleu")
        if self.Exact_Match:
            enabled.append("Exact_Match")
        
        return enabled
    
    # Callback settings
    use_sample_prediction_callback: bool = True
    sample_prediction_num_samples: int = 5
    use_memory_cleanup_callback: bool = True
    use_CSVLoggingCallback: bool = True
    
    # To continue training from latest checkpoint
    resume_from_checkpoint: bool = False