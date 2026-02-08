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
)


def format_sample(sample, prompt="Text Recognition:"):
    """Format a single sample for GLM-OCR VLM training."""
    label = sample.get("label") or sample.get("text") or sample.get("ground_truth") or ""
    image = sample.get("image")

    return {
        "images": [image] if image is not None else [],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(label)},
                ],
            },
        ],
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.dataset_name is None:
        parser.error("--dataset_name is required")

    print("Loading model...")
    model, processor = load_model(
        model_args.model_name,
        model_args.load_in_4bit,
        model_args.full_finetuning,
    )
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

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
        print(f"Auto-split: {len(train_raw)} train, {len(eval_raw)} eval")
    else:
        train_raw = raw
        print("No validation split")

    # Dry run - validate format
    if data_args.dry_run:
        print("\n--- Dry run: validating format ---")
        sample = train_raw[0]
        formatted = format_sample(sample, data_args.prompt)
        print(f"Images: {len(formatted['images'])} image(s)")
        print(f"Messages: {formatted['messages']}")

        # Test apply_chat_template with the image
        text = processor.apply_chat_template(
            formatted["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        print(f"\nChat template output:\n{text[:800]}...")

        # Check for required tokens
        required = ["[gMASK]", "<sop>", "<|user|>", "<|begin_of_image|>", "<|assistant|>"]
        missing = [t for t in required if t not in text]
        if missing:
            print(f"\nWARNING: Missing tokens: {missing}")
            print("(This might be OK if TRL injects them during collation)")
        else:
            print("\nAll required tokens present!")
        return 0

    # Use with_transform to format on-the-fly (avoids Arrow serialization issues)
    def transform_fn(batch):
        """Transform batch on-the-fly."""
        results = {"images": [], "messages": []}
        for i in range(len(batch["image"])):
            sample = {k: batch[k][i] for k in batch.keys()}
            formatted = format_sample(sample, data_args.prompt)
            results["images"].append(formatted["images"])
            results["messages"].append(formatted["messages"])
        return results

    train_dataset = train_raw.with_transform(transform_fn)
    eval_dataset = eval_raw.with_transform(transform_fn) if eval_raw else None

    print(f"Train: {len(train_raw)}, Eval: {len(eval_raw) if eval_raw else 0}")

    # Disable eval if no dataset
    if eval_dataset is None:
        training_args.eval_strategy = "no"
        training_args.load_best_model_at_end = False

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
