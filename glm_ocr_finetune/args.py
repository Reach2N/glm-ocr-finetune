"""Dataclass arguments for GLM-OCR fine-tuning."""

from dataclasses import dataclass, field
from typing import Optional

from trl import SFTConfig


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_name: str = field(
        default="zai-org/GLM-OCR",
        metadata={"help": "HuggingFace model ID or local path"},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Use 4-bit quantization (for LoRA)"},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Freeze the CogViT vision encoder (recommended)"},
    )
    freeze_multi_modal_projector: bool = field(
        default=True,
        metadata={"help": "Freeze the cross-modal MLP connector (recommended)"},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Use LoRA instead of full fine-tuning"},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank (higher = more capacity, more VRAM)"},
    )
    lora_target: str = field(
        default="all",
        metadata={"help": "LoRA target modules ('all' for all linear layers)"},
    )


@dataclass
class DataArguments:
    """Arguments for dataset configuration."""

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace dataset name"},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Name of the training split"},
    )
    eval_split: Optional[str] = field(
        default=None,
        metadata={"help": "Name of eval split (if exists in dataset)"},
    )
    eval_split_size: float = field(
        default=0.1,
        metadata={"help": "Auto-split ratio for validation if no eval_split (0 to disable)"},
    )
    prompt: str = field(
        default="Text Recognition:",
        metadata={"help": "Prompt text for OCR task"},
    )
    dry_run: bool = field(
        default=False,
        metadata={"help": "Validate format only, don't train"},
    )


@dataclass
class TrainingArguments(SFTConfig):
    """Training arguments aligned with official GLM-OCR fine-tuning guide."""

    # Output
    output_dir: str = field(
        default="./glm-ocr-finetuned",
        metadata={"help": "Output directory for checkpoints"},
    )

    # Sequence length (matches LLaMA-Factory cutoff_len)
    max_length: Optional[int] = field(
        default=2048,
        metadata={"help": "Max sequence length (matches official cutoff_len=2048)"},
    )

    # Training hyperparameters (aligned with official guide)
    num_train_epochs: float = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=2)
    learning_rate: float = field(default=1e-5)
    warmup_ratio: float = field(default=0.1)
    max_grad_norm: float = field(default=1.0)
    lr_scheduler_type: str = field(default="cosine")

    # Evaluation
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy: 'no', 'steps', 'epoch'"},
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Evaluate every N steps"},
    )

    # Saving
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Load the best model at the end of training"},
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "Metric to use for best model selection"},
    )
    greater_is_better: bool = field(default=False)

    # Logging
    logging_steps: int = field(default=10)

    # WandB tracking
    report_to: str = field(
        default="wandb",
        metadata={"help": "Report to: 'wandb', 'tensorboard', 'none'"},
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "WandB run name"},
    )

    # Other
    remove_unused_columns: bool = field(default=False)
    dataloader_num_workers: int = field(default=4)
    optim: str = field(default="adamw_torch")
