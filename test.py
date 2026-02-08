#!/usr/bin/env python3
"""Test script to evaluate GLM-OCR model on dataset samples.

Uses official GLM-OCR API from:
https://huggingface.co/zai-org/GLM-OCR
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
    """Run OCR using official GLM-OCR API pattern."""
    # Save image temporarily for URL-based API
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        image.save(f, format="PNG")
        temp_path = f.name

    try:
        # Official GLM-OCR message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": temp_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Official API: apply_chat_template with tokenize=True
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        result = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return result

    finally:
        os.unlink(temp_path)


def compute_cer(pred: str, target: str) -> float:
    """Compute Character Error Rate using edit distance."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    # Simple edit distance
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

    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    print(f"\nTesting {len(indices)} samples...\n")

    total_cer = 0.0
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        label = sample.get("label") or sample.get("text") or sample.get("ground_truth") or ""
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
