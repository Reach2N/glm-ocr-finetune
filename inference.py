#!/usr/bin/env python3
"""Inference script for GLM-OCR.

Based on official GLM-OCR usage from:
https://huggingface.co/zai-org/GLM-OCR
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
    """Run OCR on a single image using official GLM-OCR API."""
    # Build messages - official format from GLM-OCR README
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Official API: apply_chat_template with tokenize=True, return_dict=True
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Remove token_type_ids if present (as per official example)
    inputs.pop("token_type_ids", None)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)

    # Decode only generated tokens
    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return output_text


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
