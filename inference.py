#!/usr/bin/env python3
"""Inference script for GLM-OCR.

Supports base model, full fine-tuned checkpoints, and LoRA adapters.
Uses the same image processing approach as training.
"""

import argparse
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model(model_path: str, adapter_path: str = None, device_map: str = "auto"):
    """Load model and processor.

    Handles three cases:
    1. Base or full fine-tuned model: --model points to full model
    2. LoRA with explicit adapter: --model is base, --adapter is adapter dir
    3. LoRA auto-detect: --model points to dir with adapter_config.json
    """
    model_kwargs = dict(
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )

    if adapter_path:
        # Case 2: explicit adapter path
        from peft import PeftModel

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_path, **model_kwargs
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
    elif os.path.isdir(model_path) and os.path.exists(
        os.path.join(model_path, "adapter_config.json")
    ):
        # Case 3: auto-detect LoRA adapter in model_path
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_id = peft_config.base_model_name_or_path
        processor = AutoProcessor.from_pretrained(
            base_model_id, trust_remote_code=True
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_id, **model_kwargs
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        # Case 1: full model (base or fine-tuned)
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, **model_kwargs
        )

    model.eval()
    return model, processor


def run_ocr(model, processor, image_path: str, prompt: str = "Text Recognition:") -> str:
    """Run OCR using the same format as training."""
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # CRITICAL: enable_thinking=False to match training format.
    # Training template adds <think></think> before assistant content,
    # so inference must also include it via enable_thinking=False.
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = processor(
        images=[image],
        text=[text],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode only generated tokens
    input_len = inputs["input_ids"].shape[1]
    result = processor.decode(
        generated_ids[0][input_len:],
        skip_special_tokens=True,
    )

    # Strip thinking tags from output
    if "</think>" in result:
        result = result.split("</think>")[-1].strip()

    return result


def main():
    parser = argparse.ArgumentParser(description="Run OCR inference with GLM-OCR")
    parser.add_argument(
        "--model", default="zai-org/GLM-OCR", help="Model path or HuggingFace ID"
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to LoRA adapter dir (auto-detected if --model has adapter_config.json)",
    )
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--prompt", default="Text Recognition:", help="Prompt for OCR task"
    )
    parser.add_argument(
        "--device_map", default="auto", help="Device map (auto, cuda, cpu)"
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    if args.adapter:
        print(f"Loading LoRA adapter: {args.adapter}")
    model, processor = load_model(args.model, args.adapter, args.device_map)

    print(f"Running OCR on: {args.image}")
    result = run_ocr(model, processor, args.image, args.prompt)

    print("\n--- OCR Result ---")
    print(result)
    print("------------------\n")


if __name__ == "__main__":
    main()
