#!/usr/bin/env python3
"""Debug what TRL actually does during training - with custom collator fix."""

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=" * 70)
print("DEBUGGING TRL VLM TRAINING (WITH CUSTOM COLLATOR)")
print("=" * 70)

# Load model and processor
print("\n1. Loading model...")
processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR", trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    "zai-org/GLM-OCR",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Load dataset
print("\n2. Loading dataset...")
dataset = load_dataset("KiteAether/c_khmer_gemma_ocr_train", split="train[:5]")

# Format function
def format_for_vlm(sample, prompt="Text Recognition:"):
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

def transform_fn(batch):
    results = {"images": [], "messages": []}
    for i in range(len(batch["image"])):
        sample = {k: batch[k][i] for k in batch.keys()}
        formatted = format_for_vlm(sample)
        results["images"].append(formatted["images"])
        results["messages"].append(formatted["messages"])
    return results

train_dataset = dataset.with_transform(transform_fn)

# Custom collator (from train.py)
class VLMDataCollator:
    def __init__(self, processor, assistant_token_id: int):
        self.processor = processor
        self.assistant_token_id = assistant_token_id

    def __call__(self, features):
        images_list = []
        messages_list = []

        for f in features:
            images_list.append(f["images"])
            messages_list.append(f["messages"])

        all_images = []
        for imgs in images_list:
            all_images.extend(imgs)

        texts = []
        for msgs in messages_list:
            text = self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        batch = self.processor(
            images=all_images if all_images else None,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        labels = batch["input_ids"].clone()

        for i, input_ids in enumerate(batch["input_ids"]):
            assistant_positions = (input_ids == self.assistant_token_id).nonzero(as_tuple=True)[0]

            if len(assistant_positions) > 0:
                last_assistant_pos = assistant_positions[-1].item()
                labels[i, :last_assistant_pos + 1] = -100

            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                labels[i, input_ids == pad_token_id] = -100

        batch["labels"] = labels
        return batch


print("\n3. Getting assistant token ID...")
assistant_token = "<|assistant|>"
assistant_token_id = processor.tokenizer.convert_tokens_to_ids(assistant_token)
print(f"<|assistant|> token ID: {assistant_token_id}")

print("\n4. Creating custom collator...")
data_collator = VLMDataCollator(processor, assistant_token_id)

print("\n5. Creating SFTTrainer with custom collator...")
config = SFTConfig(
    output_dir="/tmp/debug",
    max_steps=1,
    per_device_train_batch_size=1,
    remove_unused_columns=False,
    max_length=None,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=processor,
    args=config,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

print(f"Trainer collator type: {type(trainer.data_collator)}")

print("\n6. Testing collator on one batch...")
dataloader = trainer.get_train_dataloader()
batch = next(iter(dataloader))

print(f"Batch keys: {list(batch.keys())}")
print(f"input_ids shape: {batch['input_ids'].shape}")

if 'pixel_values' in batch:
    print(f"pixel_values shape: {batch['pixel_values'].shape}")
    print("✓ Images ARE being passed to model!")
else:
    print("✗ pixel_values MISSING!")

if 'labels' in batch:
    print(f"labels shape: {batch['labels'].shape}")
    # Check masking
    masked = (batch['labels'] == -100).sum().item()
    non_masked = (batch['labels'] != -100).sum().item()
    total = batch['labels'].numel()
    print(f"Masked tokens (prompt): {masked}/{total} ({100*masked/total:.1f}%)")
    print(f"Non-masked tokens (response to learn): {non_masked}/{total} ({100*non_masked/total:.1f}%)")

    if masked > 0:
        print("✓ Labels ARE properly masked!")
    else:
        print("✗ Labels NOT masked - this is the bug!")

print("\n7. Visualizing the masking...")
input_ids = batch['input_ids'][0]
labels = batch['labels'][0]

# Decode tokens to see what's masked
print("\nToken-by-token breakdown:")
for i, (inp, lbl) in enumerate(zip(input_ids[:40].tolist(), labels[:40].tolist())):
    token = processor.tokenizer.decode([inp])
    status = "LEARN" if lbl != -100 else "mask"
    print(f"  {i:3d}: {status} | {inp:6d} | {repr(token)}")

if len(input_ids) > 40:
    print(f"  ... ({len(input_ids) - 40} more tokens)")

print("\n8. Forward pass test...")
try:
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
    print(f"Loss: {outputs.loss.item():.4f}")
    print("✓ Forward pass works!")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
if masked > 0 and 'pixel_values' in batch:
    print("✓ Training should now work correctly!")
    print("  - Images are being processed")
    print("  - Labels are properly masked (model learns only OCR response)")
else:
    print("✗ There's still an issue to fix")
