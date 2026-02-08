#!/usr/bin/env python3
"""Inference script for GLM-OCR.

Uses the same image processing approach as training.
"""

import argparse

import torch
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


def run_ocr(model, processor, image_path: str, prompt: str = "Text Recognition:") -> str:
    """Run OCR using the same format as training."""
    # Load image
    image = Image.open(image_path).convert("RGB")

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

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
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


def main():
    parser = argparse.ArgumentParser(description="Run OCR inference with GLM-OCR")
    parser.add_argument("--model", default="zai-org/GLM-OCR", help="Model path or HuggingFace ID")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", default="Text Recognition:", help="Prompt for OCR task")
    parser.add_argument("--device_map", default="auto", help="Device map (auto, cuda, cpu)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, processor = load_model(args.model, args.device_map)

    print(f"Running OCR on: {args.image}")
    result = run_ocr(model, processor, args.image, args.prompt)

    print("\n--- OCR Result ---")
    print(result)
    print("------------------\n")


if __name__ == "__main__":
    main()
