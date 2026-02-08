#!/usr/bin/env python3
"""Investigate what patterns exist in the training data."""

from datasets import load_dataset
from collections import Counter
import re

print("Loading dataset...")
dataset = load_dataset("KiteAether/c_khmer_gemma_ocr_train", split="train")

print(f"Dataset size: {len(dataset)}")
print(f"Columns: {dataset.column_names}")

# Check for common patterns
texts = [sample.get("text", sample.get("label", "")) for sample in dataset]

# Check text lengths
lengths = [len(t) for t in texts]
print(f"\nText length stats:")
print(f"  Min: {min(lengths)}")
print(f"  Max: {max(lengths)}")
print(f"  Avg: {sum(lengths)/len(lengths):.1f}")

# Check for very long texts (might indicate extra content)
long_texts = [(i, t) for i, t in enumerate(texts) if len(t) > 100]
print(f"\nLong texts (>100 chars): {len(long_texts)}")
if long_texts:
    print("Examples of long texts:")
    for i, t in long_texts[:5]:
        print(f"  [{i}]: {t[:150]}...")

# Check for common starting patterns
starts = Counter([t[:20] if len(t) >= 20 else t for t in texts])
print(f"\nMost common text starts:")
for pattern, count in starts.most_common(10):
    print(f"  {count:4d}x: {repr(pattern)}")

# Check for "ក្នុង" pattern (appears in predictions)
knong_count = sum(1 for t in texts if "ក្នុង" in t)
print(f"\nTexts containing 'ក្នុង': {knong_count}")

# Check for newlines in texts (might indicate multi-line labels)
newline_count = sum(1 for t in texts if "\n" in t)
print(f"Texts containing newlines: {newline_count}")

# Check for any repetitive patterns in labels
repetitive = []
for i, t in enumerate(texts):
    # Check if any 10+ char substring repeats
    for j in range(len(t) - 10):
        sub = t[j:j+10]
        if t.count(sub) > 2:
            repetitive.append((i, t))
            break

print(f"\nTexts with repetitive patterns: {len(repetitive)}")
if repetitive:
    print("Examples:")
    for i, t in repetitive[:3]:
        print(f"  [{i}]: {t[:100]}...")

# Show some random samples
print("\n\nRandom samples from dataset:")
import random
random.seed(42)
for idx in random.sample(range(len(dataset)), 10):
    t = texts[idx]
    print(f"  [{idx}]: {repr(t[:80])}")
