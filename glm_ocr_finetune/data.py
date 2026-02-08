"""Dataset formatting for GLM-OCR fine-tuning with TRL."""


def format_for_vlm(sample: dict, prompt: str = "Text Recognition:") -> dict:
    """
    Format dataset sample for TRL VLM training with explicit image placeholders.

    For GLM-OCR, we MUST include {"type": "image"} in the content,
    otherwise the model won't see the image during training.

    Args:
        sample: Dataset sample with 'image' (PIL) and 'label'/'text' columns
        prompt: OCR prompt text

    Returns:
        Dict with 'images', 'prompt', 'completion' for TRL
    """
    # Get label from various possible column names
    label = sample.get("label") or sample.get("text") or sample.get("ground_truth") or ""

    # Get image - TRL expects a list
    image = sample.get("image")
    images = [image] if image is not None else []

    # CRITICAL: Must include {"type": "image"} placeholder for VLM training
    # TRL will inject the actual PIL image from 'images' list
    return {
        "images": images,
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # <-- This is REQUIRED for GLM-OCR
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(label)},
                ],
            },
        ],
    }
