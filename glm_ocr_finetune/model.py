"""Model loading for GLM-OCR."""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model(
    model_name: str = "zai-org/GLM-OCR",
    load_in_4bit: bool = False,
    full_finetuning: bool = True,
    device_map: str = "auto",
):
    """
    Load GLM-OCR model and processor.

    Args:
        model_name: HuggingFace model ID
        load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
        full_finetuning: Enable full fine-tuning (ignored if load_in_4bit=True)
        device_map: Device placement strategy

    Returns:
        model: The loaded model
        processor: Glm46VProcessor with apply_chat_template()
    """
    # Determine dtype
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load model
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return model, processor
