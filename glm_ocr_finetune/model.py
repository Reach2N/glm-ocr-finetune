"""Model loading for GLM-OCR."""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model(
    model_name: str = "zai-org/GLM-OCR",
    load_in_4bit: bool = False,
    freeze_vision_tower: bool = True,
    freeze_multi_modal_projector: bool = True,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_target: str = "all",
    device_map: str = "auto",
):
    """
    Load GLM-OCR model and processor.

    Follows the official fine-tuning guide:
    - Vision tower (CogViT) is frozen by default
    - Multi-modal projector (MLP) is frozen by default
    - Only the language model is trained (full or LoRA)

    Args:
        model_name: HuggingFace model ID or local path
        load_in_4bit: Use 4-bit quantization (for LoRA)
        freeze_vision_tower: Freeze the CogViT vision encoder (recommended)
        freeze_multi_modal_projector: Freeze the cross-modal MLP (recommended)
        use_lora: Apply LoRA adapters instead of full fine-tuning
        lora_rank: LoRA rank (higher = more capacity, more VRAM)
        lora_target: LoRA target modules ("all" for all linear layers)
        device_map: Device placement strategy

    Returns:
        model: The loaded model (with LoRA if enabled)
        processor: Glm46VProcessor with apply_chat_template()
    """
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_kwargs = {
        "dtype": dtype,
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

    # Freeze vision tower (CogViT encoder) - recommended by official guide
    if freeze_vision_tower:
        for name, param in model.named_parameters():
            if "vision_tower" in name or "visual" in name:
                param.requires_grad = False

    # Freeze multi-modal projector (MLP connector) - recommended by official guide
    if freeze_multi_modal_projector:
        for name, param in model.named_parameters():
            if "multi_modal_projector" in name or "mlp1" in name:
                param.requires_grad = False

    # Apply LoRA if requested
    if use_lora:
        from peft import LoraConfig, get_peft_model

        if lora_target == "all":
            target_modules = "all-linear"
        else:
            target_modules = [m.strip() for m in lora_target.split(",")]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Enable gradient checkpointing for full fine-tuning memory efficiency
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return model, processor
