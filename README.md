# GLM-OCR Fine-Tune

Fine-tune [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) for OCR tasks using [TRL](https://github.com/huggingface/trl) and HuggingFace Transformers. Aligned with the [official GLM-OCR fine-tuning guide](https://huggingface.co/zai-org/GLM-OCR).

## Setup

```bash
pip install -r requirements.txt
```

| Method | Single GPU VRAM |
|---|---|
| Full fine-tuning | >= 24 GB |
| LoRA fine-tuning | >= 8 GB |

## Usage

### Full Fine-tuning (default)

Freezes vision tower + projector, trains only the language model:

```bash
python finetune.py \
  --dataset_name "SoyVitou/KhmerST-Dataset-OCR-Cropped" \
  --output_dir ./glm-ocr-finetuned \
  --num_train_epochs 3 \
  --learning_rate 1e-5
```

### LoRA Fine-tuning

Lower VRAM, faster training:

```bash
python finetune.py \
  --dataset_name "SoyVitou/KhmerST-Dataset-OCR-Cropped" \
  --output_dir ./glm-ocr-lora \
  --use_lora True \
  --lora_rank 8 \
  --load_in_4bit True
```

LoRA automatically uses `1e-4` learning rate (official recommendation).

### Dry Run

Validate dataset format without training:

```bash
python finetune.py \
  --dataset_name "SoyVitou/KhmerST-Dataset-OCR-Cropped" \
  --dry_run True
```

### Inference

```bash
python inference.py --image path/to/image.png
python inference.py --model ./glm-ocr-finetuned --image path/to/image.png
```

## Project Structure

```
glm_ocr_finetune/       # Core library
  __init__.py            # Package exports
  args.py                # CLI argument dataclasses
  data.py                # Dataset formatting for VLM training
  model.py               # Model loading, freezing, LoRA setup
  train.py               # Trainer creation with label masking
finetune.py              # CLI entry point for fine-tuning
inference.py             # CLI entry point for inference
requirements.txt         # Python dependencies
```

## Dataset Format

The training script expects a HuggingFace dataset with:
- `image`: PIL Image
- `label` (or `text` / `ground_truth`): ground truth OCR text

## Training Details

Aligned with the official GLM-OCR fine-tuning guide:

- **Layer freezing**: Vision tower (CogViT) and multi-modal projector (MLP) are frozen by default. Only the language model is trained.
- **Label masking**: Custom `VLMDataCollator` masks all tokens up to `</think>` so the model only learns to predict OCR text.
- **Defaults**: `lr=1e-5`, `cosine` scheduler, `batch_size=4`, `grad_accum=2`, `cutoff_len=2048`.
- **LoRA**: PEFT LoRA with `rank=8`, `target=all-linear`, `lr=1e-4`.
- **Eval**: Automatic train/eval split (`--eval_split_size 0.1`).
- **Logging**: WandB (`--report_to wandb`), or `--report_to none` to disable.

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--use_lora` | False | Use LoRA instead of full fine-tuning |
| `--lora_rank` | 8 | LoRA rank |
| `--load_in_4bit` | False | 4-bit quantization (for LoRA) |
| `--freeze_vision_tower` | True | Freeze CogViT encoder |
| `--freeze_multi_modal_projector` | True | Freeze MLP connector |
| `--learning_rate` | 1e-5 | 1e-5 full, auto 1e-4 for LoRA |
| `--per_device_train_batch_size` | 4 | Batch size per GPU |
| `--gradient_accumulation_steps` | 2 | Effective batch = 4x2 = 8 |

## License

See [GLM-OCR model card](https://huggingface.co/zai-org/GLM-OCR) for model license terms.
