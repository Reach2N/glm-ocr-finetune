"""Dataset formatting for GLM-OCR fine-tuning with TRL."""


def format_for_vlm(sample: dict, prompt: str = "Text Recognition:") -> dict:
    """
    Format dataset sample for TRL VLM training.

    For GLM-OCR, we MUST include {"type": "image"} in the content,
    otherwise the model won't see the image during training.

    The GLM-OCR chat template automatically adds <think></think>
    before assistant responses, so we don't need to add it manually.

    Args:
        sample: Dataset sample with 'image' (PIL) and 'label'/'text' columns
        prompt: OCR prompt text

    Returns:
        Dict with 'images' and 'messages' for TRL SFTTrainer
    """
    # Get label from various possible column names
    label = sample.get("label") or sample.get("text") or sample.get("ground_truth") or ""

    # Get image - TRL expects a list
    image = sample.get("image")
    images = [image] if image is not None else []

    # Return format compatible with TRL SFTTrainer
    # CRITICAL: Must include {"type": "image"} placeholder for VLM training
    return {
        "images": images,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Required for GLM-OCR to see the image
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
