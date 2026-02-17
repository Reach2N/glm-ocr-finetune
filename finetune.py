#!/usr/bin/env python3
"""CLI entry point for GLM-OCR fine-tuning.

Based on official TRL VLM training patterns:
https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import HfArgumentParser

from glm_ocr_finetune import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    load_model,
    create_trainer,
    format_for_vlm,
)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.dataset_name is None:
        parser.error("--dataset_name is required")

    # Override learning rate for LoRA if user didn't set it explicitly
    # Official guide: 1e-4 for LoRA, 1e-5 for full
    if model_args.use_lora and training_args.learning_rate == 1e-5:
        training_args.learning_rate = 1e-4

    print("Loading model...")
    model, processor = load_model(
        model_args.model_name,
        load_in_4bit=model_args.load_in_4bit,
        freeze_vision_tower=model_args.freeze_vision_tower,
        freeze_multi_modal_projector=model_args.freeze_multi_modal_projector,
        use_lora=model_args.use_lora,
        lora_rank=model_args.lora_rank,
        lora_target=model_args.lora_target,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded. Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

    # Load dataset
    print(f"Loading dataset: {data_args.dataset_name}")
    raw = load_dataset(data_args.dataset_name, split=data_args.train_split)

    # Split for validation
    eval_raw = None
    if data_args.eval_split:
        train_raw = raw
        eval_raw = load_dataset(data_args.dataset_name, split=data_args.eval_split)
        print(f"Using eval split: {data_args.eval_split}")
    elif data_args.eval_split_size > 0:
        splits = raw.train_test_split(test_size=data_args.eval_split_size, seed=42)
        train_raw, eval_raw = splits["train"], splits["test"]
        # Cap eval set to avoid slow evaluation on large datasets
        max_eval = 1000
        if len(eval_raw) > max_eval:
            eval_raw = eval_raw.select(range(max_eval))
        print(f"Auto-split: {len(train_raw)} train, {len(eval_raw)} eval")
    else:
        train_raw = raw
        print("No validation split")

    # Dry run - validate format
    if data_args.dry_run:
        print("\n--- Dry run: validating format ---")
        sample = train_raw[0]
        formatted = format_for_vlm(sample, data_args.prompt)
        print(f"Images: {len(formatted['images'])} image(s)")
        print(f"Messages: {formatted['messages']}")

        text = processor.apply_chat_template(
            formatted["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        print(f"\nChat template output:\n{text[:800]}...")

        required = ["[gMASK]", "<sop>", "<|user|>", "<|begin_of_image|>", "<|assistant|>"]
        missing = [t for t in required if t not in text]
        if missing:
            print(f"\nWARNING: Missing tokens: {missing}")
        else:
            print("\nAll required tokens present!")
        return 0

    # Use with_transform to format on-the-fly (avoids Arrow serialization issues)
    def transform_fn(batch):
        results = {"images": [], "messages": []}
        for i in range(len(batch["image"])):
            sample = {k: batch[k][i] for k in batch.keys()}
            formatted = format_for_vlm(sample, data_args.prompt)
            results["images"].append(formatted["images"])
            results["messages"].append(formatted["messages"])
        return results

    train_dataset = train_raw.with_transform(transform_fn)
    eval_dataset = eval_raw.with_transform(transform_fn) if eval_raw else None

    print(f"Train: {len(train_raw)}, Eval: {len(eval_raw) if eval_raw else 0}")

    # Disable eval if no dataset, enable best-model tracking if eval exists
    if eval_dataset is None or training_args.eval_strategy == "no":
        training_args.eval_strategy = "no"
        training_args.load_best_model_at_end = False
        eval_dataset = None
    else:
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False

    # Create trainer
    trainer = create_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        training_args=training_args,
        eval_dataset=eval_dataset,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    print("Done.")


if __name__ == "__main__":
    exit(main() or 0)
