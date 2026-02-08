#!/usr/bin/env python3
"""Debug script to verify training and inference formats match.

Run this on your server to compare the exact format used in training vs inference.
"""

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained('zai-org/GLM-OCR', trust_remote_code=True)

print("=" * 70)
print("GLM-OCR FORMAT COMPARISON: Training vs Inference")
print("=" * 70)

# ============================================================
# TRAINING FORMAT (from finetune.py format_sample)
# ============================================================
print("\n1. TRAINING FORMAT (with assistant response)")
print("-" * 70)

training_messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Text Recognition:"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Hello World"},
        ],
    },
]

training_text = processor.apply_chat_template(
    training_messages,
    tokenize=False,
    add_generation_prompt=False,
)
print(training_text)
print()

# ============================================================
# INFERENCE FORMAT (OLD - without enable_thinking)
# ============================================================
print("\n2. INFERENCE FORMAT - OLD (enable_thinking NOT set)")
print("-" * 70)

inference_messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Text Recognition:"},
        ],
    },
]

inference_text_old = processor.apply_chat_template(
    inference_messages,
    tokenize=False,
    add_generation_prompt=True,
    # enable_thinking NOT set - this was the bug!
)
print(inference_text_old)
print()

# ============================================================
# INFERENCE FORMAT (NEW - with enable_thinking=False)
# ============================================================
print("\n3. INFERENCE FORMAT - NEW (enable_thinking=False)")
print("-" * 70)

inference_text_new = processor.apply_chat_template(
    inference_messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # This matches training format!
)
print(inference_text_new)
print()

# ============================================================
# ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

# Extract the format after <|assistant|>
training_after_assistant = training_text.split("<|assistant|>")[1] if "<|assistant|>" in training_text else ""
old_after_assistant = inference_text_old.split("<|assistant|>")[1] if "<|assistant|>" in inference_text_old else ""
new_after_assistant = inference_text_new.split("<|assistant|>")[1] if "<|assistant|>" in inference_text_new else ""

print(f"\nTraining format after <|assistant|>:")
print(f"  {repr(training_after_assistant[:100])}")

print(f"\nOLD inference format after <|assistant|>:")
print(f"  {repr(old_after_assistant[:100])}")

print(f"\nNEW inference format after <|assistant|>:")
print(f"  {repr(new_after_assistant[:100])}")

# Check for <think></think>
train_has_think = "<think></think>" in training_text
old_has_think = "<think></think>" in inference_text_old
new_has_think = "<think></think>" in inference_text_new

print(f"\n<think></think> present:")
print(f"  Training:       {'YES ✓' if train_has_think else 'NO ✗'}")
print(f"  Inference OLD:  {'YES ✓' if old_has_think else 'NO ✗'}")
print(f"  Inference NEW:  {'YES ✓' if new_has_think else 'NO ✗'}")

if train_has_think and new_has_think:
    print("\n✓ Training and NEW inference format MATCH!")
elif train_has_think and not new_has_think:
    print("\n✗ MISMATCH: Training has <think></think> but inference doesn't!")
else:
    print("\n? Unexpected format - please check manually")
