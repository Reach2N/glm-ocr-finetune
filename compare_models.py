#!/usr/bin/env python3
"""Compare BASE model vs finetuned model on same samples."""

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset
import random

def run_ocr(model, processor, image, prompt="Text Recognition:"):
    """Run OCR inference."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

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
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.2,
        )

    input_len = inputs["input_ids"].shape[1]
    result = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    return result


print("Loading BASE model...")
base_processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR", trust_remote_code=True)
base_model = AutoModelForImageTextToText.from_pretrained(
    "zai-org/GLM-OCR",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
base_model.eval()

print("Loading FINETUNED model...")
ft_processor = AutoProcessor.from_pretrained("./glm-ocr-finetuned", trust_remote_code=True)
ft_model = AutoModelForImageTextToText.from_pretrained(
    "./glm-ocr-finetuned",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
ft_model.eval()

print("Loading dataset...")
dataset = load_dataset("KiteAether/c_khmer_gemma_ocr_train", split="train")

random.seed(42)
indices = random.sample(range(len(dataset)), 5)

print("\n" + "=" * 70)
print("COMPARISON: BASE vs FINETUNED")
print("=" * 70)

for idx in indices:
    sample = dataset[idx]
    image = sample["image"]
    gt = sample.get("text", sample.get("label", ""))

    base_pred = run_ocr(base_model, base_processor, image)
    ft_pred = run_ocr(ft_model, ft_processor, image)

    print(f"\n[Sample {idx}]")
    print(f"  GT:       {gt[:60]}{'...' if len(gt) > 60 else ''}")
    print(f"  BASE:     {base_pred[:60]}{'...' if len(base_pred) > 60 else ''}")
    print(f"  FINETUNE: {ft_pred[:60]}{'...' if len(ft_pred) > 60 else ''}")
