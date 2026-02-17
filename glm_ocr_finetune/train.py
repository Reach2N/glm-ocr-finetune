"""Training utilities for GLM-OCR."""

import torch
from trl import SFTTrainer

from .args import TrainingArguments


class VLMDataCollator:
    """
    Data collator for VLM training that properly masks prompt tokens in labels.

    TRL's default collator doesn't mask labels correctly for GLM-OCR,
    causing the model to learn to predict prompts instead of just responses.
    """

    def __init__(self, processor, assistant_token_id: int, think_token_id: int, think_end_token_id: int):
        self.processor = processor
        self.assistant_token_id = assistant_token_id
        self.think_token_id = think_token_id
        self.think_end_token_id = think_end_token_id

    def __call__(self, features):
        # Separate images and messages
        images_list = []
        messages_list = []

        for f in features:
            images_list.append(f["images"])
            messages_list.append(f["messages"])

        # Flatten images (each sample has a list of images)
        all_images = []
        for imgs in images_list:
            all_images.extend(imgs)

        # Apply chat template to get text
        texts = []
        for msgs in messages_list:
            text = self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        # Process with processor (handles both text and images)
        batch = self.processor(
            images=all_images if all_images else None,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Create labels with proper masking
        labels = batch["input_ids"].clone()

        for i, input_ids in enumerate(batch["input_ids"]):
            # Find the position of </think> token - mask everything up to and including it
            # This ensures the model only learns to generate actual OCR text
            think_end_positions = (input_ids == self.think_end_token_id).nonzero(as_tuple=True)[0]

            if len(think_end_positions) > 0:
                # Mask everything up to and including </think>
                # Also mask the newline after </think>
                last_think_end_pos = think_end_positions[-1].item()
                # Mask up to </think> + 1 (for the newline after it)
                mask_end = min(last_think_end_pos + 2, len(input_ids))
                labels[i, :mask_end] = -100
            else:
                # Fallback: mask up to <|assistant|>
                assistant_positions = (input_ids == self.assistant_token_id).nonzero(as_tuple=True)[0]
                if len(assistant_positions) > 0:
                    last_assistant_pos = assistant_positions[-1].item()
                    labels[i, :last_assistant_pos + 1] = -100

            # Also mask padding tokens
            if hasattr(self.processor, 'tokenizer'):
                pad_token_id = self.processor.tokenizer.pad_token_id
                if pad_token_id is not None:
                    labels[i, input_ids == pad_token_id] = -100

        batch["labels"] = labels

        return batch


def create_trainer(
    model,
    processor,
    train_dataset,
    training_args: TrainingArguments,
    eval_dataset=None,
):
    """
    Create SFTTrainer for GLM-OCR fine-tuning.

    Uses custom VLMDataCollator that properly masks prompt tokens in labels.

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

    # Get token IDs for label masking
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|assistant|>")
    think_token_id = processor.tokenizer.convert_tokens_to_ids("<think>")
    think_end_token_id = processor.tokenizer.convert_tokens_to_ids("</think>")

    # Create custom collator that properly masks labels
    # Masks everything up to and including </think> so model only learns OCR text
    data_collator = VLMDataCollator(processor, assistant_token_id, think_token_id, think_end_token_id)

    return SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
