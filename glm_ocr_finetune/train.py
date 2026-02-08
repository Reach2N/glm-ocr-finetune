"""Training utilities for GLM-OCR."""

import torch
from trl import SFTTrainer, SFTConfig

from .args import TrainingArguments


def create_trainer(
    model,
    processor,
    train_dataset,
    training_args: TrainingArguments,
    eval_dataset=None,
):
    """
    Create SFTTrainer for GLM-OCR fine-tuning.

    Uses TRL's built-in VLM support - no custom collator needed.
    TRL automatically detects VLM from processor and uses DataCollatorForVisionLanguageModeling.

    Args:
        model: The loaded model
        processor: The processor (Glm46VProcessor)
        train_dataset: Training dataset with 'images' and 'messages' columns
        training_args: TrainingArguments instance
        eval_dataset: Optional validation dataset

    Returns:
        SFTTrainer ready for training
    """
    # Set precision based on hardware support if not set
    if training_args.bf16 is None and training_args.fp16 is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            training_args.bf16 = True
            training_args.fp16 = False
        else:
            training_args.bf16 = False
            training_args.fp16 = True

    # TRL handles VLM collation internally when it detects a ProcessorMixin
    # Key: pass processing_class, not tokenizer
    # Key: set max_length=None to avoid truncating image tokens
    return SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # No data_collator - TRL uses DataCollatorForVisionLanguageModeling automatically
    )
