# GLM-OCR Fine-Tune

Fine-tune [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) for OCR tasks using [TRL](https://github.com/huggingface/trl) and HuggingFace Transformers.

All training hyperparameters are aligned with the [official GLM-OCR fine-tuning guide](https://github.com/zai-org/GLM-OCR/tree/main/examples/finetune).

## Requirements

- Python 3.10+
- CUDA-capable GPU

| Method | Single GPU VRAM |
|---|---|
| Full fine-tuning | >= 24 GB |
| LoRA (4-bit) | >= 8 GB |

## Installation

```bash
git clone https://github.com/Reach2N/glm-ocr-finetune.git
cd glm-ocr-finetune
pip install -r requirements.txt
```

## Quick Start

### Full Fine-tuning

Freezes vision tower + projector, trains only the language model (0.5B params):

```bash
python finetune.py \
  --dataset_name "your-hf-dataset" \
  --output_dir ./glm-ocr-finetuned \
  --num_train_epochs 3 \
  --report_to none
```

### LoRA Fine-tuning

Lower VRAM, faster training. Learning rate auto-adjusts to `1e-4` (official recommendation):

```bash
python finetune.py \
  --dataset_name "your-hf-dataset" \
  --output_dir ./glm-ocr-lora \
  --use_lora True \
  --lora_rank 8 \
  --load_in_4bit True \
  --report_to none
```

### Quick Test Run

Run a short training test to verify everything works:

```bash
python finetune.py \
  --dataset_name "your-hf-dataset" \
  --output_dir ./test-run \
  --max_steps 100 \
  --eval_strategy no \
  --report_to none
```

### Dry Run

Validate dataset format without training:

```bash
python finetune.py \
  --dataset_name "your-hf-dataset" \
  --dry_run True
```

## Inference

Works with base model, full fine-tuned checkpoints, and LoRA adapters:

```bash
# Base model
python inference.py --image path/to/image.png

# Full fine-tuned
python inference.py --model ./glm-ocr-finetuned --image path/to/image.png

# LoRA adapter (auto-detected from adapter_config.json)
python inference.py --model ./glm-ocr-lora --image path/to/image.png

# LoRA with explicit base + adapter
python inference.py --model zai-org/GLM-OCR --adapter ./glm-ocr-lora --image path/to/image.png
```

## Dataset Format

The training script expects a HuggingFace dataset with:

- `image` - PIL Image
- `label` (or `text` / `ground_truth`) - ground truth OCR text

Any HuggingFace dataset with these columns works out of the box. The script auto-detects the label column name.

Example compatible datasets:
- `SoyVitou/KhmerST-Dataset-OCR-Cropped`
- `SoyVitou/KhmerSynthetic1M`

## Architecture

```
GLM-OCR (1.1B total)
├── Vision Tower (CogViT)           <- Frozen (default)
├── Multi-Modal Projector (MLP)     <- Frozen (default)
└── Language Model (GLM-0.5B)       <- Trained
```

### How Training Works

1. **Image** is processed by the frozen CogViT vision encoder into visual tokens
2. Visual tokens are projected into the language model's embedding space by the frozen MLP
3. The language model learns to generate OCR text conditioned on the visual tokens
4. **Label masking**: Everything up to `</think>` is masked so the model only learns to predict OCR text, not the prompt

The `</think>` token comes from GLM-OCR's built-in [chat template](https://huggingface.co/zai-org/GLM-OCR/blob/main/chat_template.jinja) which always inserts `<think></think>` before assistant responses.

### Training vs Inference Alignment

Both training and inference use the same pipeline:

| Step | Training | Inference |
|---|---|---|
| Chat template | `processor.apply_chat_template()` | `processor.apply_chat_template()` |
| Thinking mode | Template inserts `<think></think>` | `enable_thinking=False` (same result) |
| Image processing | `processor(images=..., text=...)` | `processor(images=..., text=...)` |
| Message format | `[{"type": "image"}, {"type": "text", "text": "Text Recognition:"}]` | Same |

## CLI Reference

### Model Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `zai-org/GLM-OCR` | HuggingFace model ID or local path |
| `--use_lora` | `False` | Use LoRA instead of full fine-tuning |
| `--lora_rank` | `8` | LoRA rank (higher = more capacity, more VRAM) |
| `--lora_target` | `all` | LoRA target modules |
| `--load_in_4bit` | `False` | 4-bit quantization (for LoRA) |
| `--freeze_vision_tower` | `True` | Freeze CogViT vision encoder |
| `--freeze_multi_modal_projector` | `True` | Freeze cross-modal MLP connector |

### Data Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | required | HuggingFace dataset name |
| `--train_split` | `train` | Training split name |
| `--eval_split` | `None` | Eval split name (if dataset has one) |
| `--eval_split_size` | `0.1` | Auto-split ratio for validation (0 to disable) |
| `--prompt` | `Text Recognition:` | Prompt text for OCR task |
| `--dry_run` | `False` | Validate format only, don't train |

### Training Arguments

All [TRL SFTConfig](https://huggingface.co/docs/trl/sft_trainer#trl.SFTConfig) arguments are supported. Key defaults aligned with the official guide:

| Argument | Default | Description |
|---|---|---|
| `--output_dir` | `./glm-ocr-finetuned` | Output directory |
| `--num_train_epochs` | `3` | Number of training epochs |
| `--max_steps` | `-1` | Max steps (overrides epochs if set) |
| `--learning_rate` | `1e-5` | LR (auto `1e-4` for LoRA) |
| `--lr_scheduler_type` | `cosine` | LR scheduler |
| `--warmup_ratio` | `0.1` | Warmup ratio |
| `--per_device_train_batch_size` | `4` | Batch size per GPU |
| `--gradient_accumulation_steps` | `2` | Gradient accumulation (effective batch = 8) |
| `--max_length` | `2048` | Max sequence length |
| `--eval_strategy` | `steps` | Eval strategy: `no`, `steps`, `epoch` |
| `--eval_steps` | `100` | Evaluate every N steps |
| `--save_steps` | `500` | Save checkpoint every N steps |
| `--save_total_limit` | `3` | Max checkpoints to keep |
| `--report_to` | `wandb` | Logging: `wandb`, `tensorboard`, `none` |
| `--max_grad_norm` | `1.0` | Gradient clipping |

### Inference Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `zai-org/GLM-OCR` | Model path or HuggingFace ID |
| `--adapter` | `None` | LoRA adapter path (auto-detected if model dir has `adapter_config.json`) |
| `--image` | required | Path to image file |
| `--prompt` | `Text Recognition:` | Prompt for OCR task |
| `--device_map` | `auto` | Device map |

## Project Structure

```
glm_ocr_finetune/       # Core library
  __init__.py            # Package exports
  args.py                # CLI argument dataclasses
  data.py                # Dataset formatting for VLM training
  model.py               # Model loading, freezing, LoRA setup
  train.py               # Trainer with custom VLMDataCollator
finetune.py              # CLI entry point for fine-tuning
inference.py             # CLI entry point for inference
requirements.txt         # Dependencies
```

## Official Config Comparison

Our defaults match the [official GLM-OCR fine-tuning configs](https://github.com/zai-org/GLM-OCR/tree/main/examples/finetune):

| Parameter | Official (Full) | Official (LoRA) | Ours |
|---|---|---|---|
| freeze_vision_tower | true | true | true |
| freeze_multi_modal_projector | true | true | true |
| learning_rate | 1e-5 | 1e-4 | 1e-5 (auto 1e-4 LoRA) |
| lr_scheduler | cosine | cosine | cosine |
| batch_size | 4 | 4 | 4 |
| grad_accum | 2 | 4 | 2 |
| cutoff_len | 2048 | 2048 | 2048 |
| lora_rank | - | 8 | 8 |
| lora_target | - | all | all-linear |
| warmup_ratio | 0.1 | 0.1 | 0.1 |
| epochs | 3 | 3 | 3 |

## License

See [GLM-OCR model card](https://huggingface.co/zai-org/GLM-OCR) for model license terms.
