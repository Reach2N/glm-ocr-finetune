# GLM-OCR Fine-Tune

Fine-tune [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) for OCR tasks using [TRL](https://github.com/huggingface/trl) and HuggingFace Transformers.

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU with sufficient VRAM (24GB+ recommended for full fine-tuning).

## Usage

### Fine-tuning

```bash
python finetune.py \
  --dataset_name "SoyVitou/KhmerST-Dataset-OCR-Cropped" \
  --output_dir ./glm-ocr-finetuned \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5
```

**Dry run** (validate dataset format without training):

```bash
python finetune.py \
  --dataset_name "SoyVitou/KhmerST-Dataset-OCR-Cropped" \
  --dry_run True
```

### Inference

Run OCR on an image using the base or fine-tuned model:

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
  model.py               # Model/processor loading
  train.py               # Trainer creation with label masking
finetune.py              # CLI entry point for fine-tuning
inference.py             # CLI entry point for inference
requirements.txt         # Python dependencies
```

## Dataset Format

The training script expects a HuggingFace dataset with:
- `image`: PIL Image
- `label` (or `text` / `ground_truth`): ground truth OCR text

## Key Training Details

- Uses TRL `SFTTrainer` with a custom `VLMDataCollator` that masks prompt tokens in labels, so the model only learns to predict OCR text (not the prompt or `<think></think>` tags).
- Automatic train/eval split (configurable via `--eval_split_size`).
- Supports WandB logging (`--report_to wandb`).
- Mixed precision (bf16/fp16) auto-detected.

## License

See [GLM-OCR model card](https://huggingface.co/zai-org/GLM-OCR) for model license terms.
