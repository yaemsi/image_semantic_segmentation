## Image Semantic Segmentation

This repository contains a training pipeline for semantic segmentation: given an input RGB image, the model predicts a per-pixel mask indicating whether the logo overlaps each pixel.

The original challenge setting and dataset description are summarized in [`documents/instructions.pdf`](documents/instructions.pdf).

## What is implemented

- Dataset preprocessing from a provided zip into a Hugging Face `datasets` format (saved to disk).
- Training of a U-Net++-style model (via `segmentation-models-pytorch`) using Hugging Face `Trainer`.
- Metric computation using mean IoU (`evaluate.load("mean_iou")`).
- Model checkpoint saving to `output_dir/final`.
- A benchmark script (`image_semantic_segmentation/evaluation.py`) that computes IoU from saved predictions.

Notes:
- `main.py --evaluate` is currently `NotImplementedError`.
- `evaluation.py` expects an `infer.py` script to exist (it defaults to `./infer.py`), but `infer.py` is not included in this repo.

## Quick start

### 1) (Optional) Preprocess the raw dataset zip

The preprocessing pipeline expects extracted data to have:

- `img/`: RGB images
- `mask/`: corresponding masks (BMP in the original challenge)

Example (using the included `data/data.zip`):

```bash
python main.py --preprocess \
  --extract_data \
  --zip_file_path ./data/data.zip \
  --img_dir ./data/extracted \
  --process_data \
  --ds_dir ./data/processed
```

Alternatively, you can skip preprocessing if `./data/processed` already exists.

### 2) Train

The repo includes a ready-to-run script (`launch.sh`). It launches training with `accelerate` and the config in `configs/acc_no_dynamo.yaml`.

```bash
bash launch.sh
```

You can also run the same command directly:

```bash
accelerate launch --config_file ./configs/acc_no_dynamo.yaml main.py --train \
  --dataset_path ./data/processed \
  --output_dir ./output_image_segmentation \
  --learning_rate 6e-5 \
  --max_steps 25000 \
  --warmup_steps 8000 \
  --save_steps 2500 \
  --eval_steps 2500 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32
```

The script saves the final model to:

- `./output_image_segmentation/final/`

During training, checkpoints may also be created under `output_dir/checkpoint-*` depending on the `TrainingArguments` save strategy.

## CLI reference (`main.py`)

`main.py` uses Hugging Face `HfArgumentParser` with these mutually exclusive top-level modes:

- `--preprocess`: extract and/or convert the dataset
- `--train`: load a processed dataset from disk and train the model
- `--evaluate`: not implemented yet

### Preprocess arguments

- `--extract_data` + `--zip_file_path` + `--img_dir`
- `--process_data` + `--img_dir` + `--ds_dir`

### Train arguments (key ones)

- `--dataset_path`: path to the dataset saved by `build_dataset()` (e.g. `./data/processed`)
- `--output_dir`: where checkpoints and `final` are written
- Common optimization settings:
  - `--learning_rate`
  - `--max_steps`
  - `--warmup_steps`
  - `--save_steps`
  - `--eval_steps`
  - `--per_device_train_batch_size`
  - `--per_device_eval_batch_size`

Model hyperparameters are controlled by `ModelArguments` (encoder/decoder channels, etc.). Defaults are set in `image_semantic_segmentation/arguments.py`.

## Model and training details

- Model: `segmentation_models_pytorch.UnetPlusPlus`
- Number of classes: `2` (background vs. logo)
- Loss: in `image_semantic_segmentation/training.py`, training uses `CombinedLoss`, currently implemented as BCE on the probability of the logo class (computed from `softmax(logits)`), with padding removal handled in the loss function.
- Metric/loss padding handling: images are padded to a multiple of 32, and then evaluation/loss slice out the padded rows using fixed indices (tied to the original image height used by the challenge).
- Metrics: mean IoU via the `evaluate` package.

## Evaluation / inference

The challenge expects a script:

```bash
python infer.py <image_dir> <output_dir>
```

This repo does not include `infer.py`, but it does include a benchmark script:

- `image_semantic_segmentation/evaluation.py`

What `evaluation.py` expects:

- `--script_path`: path to your inference script (default `./infer.py`)
- `--testset_path`: directory containing one or more test-set folders (default `./test/`)
- Each test-set folder contains:
  - `img/`: input images
  - `mask/`: ground-truth masks (`.bmp`)
- Your `infer.py` should write predictions under:
  - `<prediction_path>/<test_name>/<same_filename_as_img>`
- The evaluator thresholds prediction pixels with `pred > 0.5` (so predictions should be binary masks where logo pixels are non-zero, commonly `0/255` grayscale).

Example benchmark command:

```bash
python -m image_semantic_segmentation.evaluation \
  --script_path ./infer.py \
  --testset_path ./data/test \
  --prediction_path ./output_predictions
```

## Repository layout

- `main.py`: single entrypoint (preprocess/train)
- `launch.sh`: example `accelerate` training launcher
- `configs/`: `accelerate` configuration files
- `image_semantic_segmentation/`: model, dataset processing, training, evaluation
- `data/`: raw zip, extracted images, and processed Hugging Face datasets
- `output_image_segmentation/`: training outputs/checkpoints
- `documents/instructions.pdf`: challenge description and deliverables

## Dependencies

Dependencies are declared in `pyproject.toml` (including `torch`, `segmentation-models-pytorch`, `albumentations`, `datasets`, `transformers`, `evaluate`, `wandb`).

`pyproject.toml` pins a CUDA-enabled PyTorch build (see `tool.uv.sources`).

## Important notes

- `launch.sh` sets a `WANDB_API_KEY` value. Replace it with your own and avoid committing real credentials.
- If you want to disable W&B logging, you may need to modify the training code in `main.py` because `wandb.init(...)` is called unconditionally.

