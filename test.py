#!/usr/bin/env python3
"""Test script to evaluate GLM-OCR model on dataset samples.

Uses the same image processing approach as training.
"""

import argparse
import random

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model(model_path: str, device_map: str = "auto"):
    """Load model and processor."""
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def run_ocr(model, processor, image: Image.Image, prompt: str = "Text Recognition:") -> str:
    """Run OCR using the same format as training."""
    # Build messages with image placeholder (same as training)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Get text with image placeholders
    # CRITICAL: enable_thinking=False to match training format
    # Training adds <think></think> before assistant content,
    # so inference must also include it via enable_thinking=False
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Process text AND image together (same as training collator)
    inputs = processor(
        images=[image],
        text=[text],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # Generate with repetition penalty to prevent loops
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,  # OCR text is usually short
            do_sample=False,
            repetition_penalty=1.2,  # Penalize repetition
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode only generated tokens
    input_len = inputs["input_ids"].shape[1]
    result = processor.decode(
        generated_ids[0][input_len:],
        skip_special_tokens=True,
    )

    # Clean up thinking tags
    if "</think>" in result:
        result = result.split("</think>")[-1].strip()

    return result


def compute_cer(pred: str, target: str) -> float:
    """Compute Character Error Rate using edit distance."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    m, n = len(target), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if target[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n] / m


def main():
    parser = argparse.ArgumentParser(description="Test GLM-OCR on dataset samples")
    parser.add_argument("--model", default="zai-org/GLM-OCR", help="Model path or HuggingFace ID")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--prompt", default="Text Recognition:", help="OCR prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, processor = load_model(args.model)

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)

    # Detect column names
    text_col = "text" if "text" in dataset.column_names else "label"
    image_col = "image" if "image" in dataset.column_names else "images"
    print(f"Using columns: image='{image_col}', text='{text_col}'")

    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    print(f"\nTesting {len(indices)} samples...\n")

    total_cer = 0.0
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample[image_col]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        label = sample.get(text_col, "")
        pred = run_ocr(model, processor, image, args.prompt)
        cer = compute_cer(pred, label)
        total_cer += cer

        print(f"[{i+1}/{len(indices)}] CER: {cer:.2%}")
        print(f"  GT:   {label[:80]}{'...' if len(label) > 80 else ''}")
        print(f"  Pred: {pred[:80]}{'...' if len(pred) > 80 else ''}")
        print()

    avg_cer = total_cer / len(indices)
    print("=" * 60)
    print(f"Average CER: {avg_cer:.2%}")
    print(f"Accuracy: {1 - avg_cer:.2%}")


if __name__ == "__main__":
    main()
