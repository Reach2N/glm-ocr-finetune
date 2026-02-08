#!/usr/bin/env python3
"""Test BASE GLM-OCR model to verify our inference format works."""

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset

# Load BASE model (not finetuned)
print("Loading BASE model: zai-org/GLM-OCR")
processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR", trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    "zai-org/GLM-OCR",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

# Load one sample
print("Loading dataset sample...")
dataset = load_dataset("KiteAether/c_khmer_gemma_ocr_train", split="train")
sample = dataset[0]
image = sample["image"]
label = sample.get("text", "")

print(f"Ground truth: {label}")

# Test inference
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Text Recognition:"},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print(f"\nPrompt format:\n{text}\n")

inputs = processor(
    images=[image],
    text=[text],
    return_tensors="pt",
    padding=True,
).to(model.device)

print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Pixel values shape: {inputs.get('pixel_values', 'MISSING').shape if 'pixel_values' in inputs else 'MISSING!'}")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

input_len = inputs["input_ids"].shape[1]
result = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)

print(f"\nBASE model prediction: {result}")
print(f"Ground truth: {label}")
