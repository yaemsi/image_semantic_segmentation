## Image Semantic Segmentation

This project trains and evaluates a semantic segmentation model to predict, for every pixel, whether the **Mila logo overlaps** the pixel. The challenge setting and dataset details are described in [`documents/instructions.pdf`](documents/instructions.pdf).

## Goals

Given an RGB image (`340x512`), the model outputs a binary mask of the same spatial size indicating logo overlap for each pixel. Performance is measured with **IoU** (mean Intersection over Union).

## Model

- Architecture: `segmentation_models_pytorch.UnetPlusPlus` wrapped as a Hugging Face `PreTrainedModel`.
- Backbone: `resnet34` by default.
- Classes: `2` (background vs. logo).
- Loss/metrics:
  - Loss: `BCEWithLogitsLoss` on the **logo channel** (`logits[:, 1, ...]`) with a padding crop to remove padded rows/cols.
  - Metric: `evaluate.load("mean_iou")` computed after taking `argmax` and removing the same padding via fixed indices.

Padding constants are hard-coded in `image_semantic_segmentation/dataset.py` / `image_semantic_segmentation/model.py`:
`IMAGE_H=340`, `PAD_H=6`, `IMAGE_W=512`, `PAD_W=0`.

## Repository layout

- `main.py`: single entrypoint (preprocess/train/evaluate/predict) driven by CLI flags.
- `image_semantic_segmentation/`: dataset building, model, training utilities, and inference utilities.
  - `image_semantic_segmentation/dataset.py`: preprocessing + padding/collation
  - `image_semantic_segmentation/model.py`: UNet++ wrapper
  - `image_semantic_segmentation/training.py`: loss + metric functions
  - `image_semantic_segmentation/inference.py`: evaluation + prediction helpers
- `scripts/`: runnable shell wrappers
  - `process_data.sh`, `train.sh`, `evaluate.sh`, `predict.sh`
- `data/`: extracted dataset + Hugging Face processed dataset
- `output_image_segmentation/`: training checkpoints and final model
- `output_evaluation/`: evaluation metrics json
- `output_prediction/`: predicted masks (`.bmp`) and a small summary json

## Dataset format

The preprocessing assumes your raw data folder contains:

- `img/`: RGB images (JPG)
- `mask/`: corresponding masks (BMP)

The zip extraction step simply unpacks the provided archive into `img_dir`, and the code expects the extracted structure above.

After preprocessing, the repo produces:

- `data/processed/`: Hugging Face `datasets` dataset (contains `train`, `validation`, and `test`)
- `data/raw/test/`: a raw `img/` + `mask/` subset corresponding to the test split

## How to run

### 0) (Optional) Prepare environment

Dependencies are defined in `pyproject.toml` / `uv.lock`. If you use `uv`, a typical workflow is:

```bash
uv sync --frozen
```

### 1) Preprocess the dataset

Runs `main.py --preprocess` and builds the Hugging Face dataset on disk.

```bash
bash scripts/process_data.sh
```

This uses:

- `data/data.zip` as input
- writes to `data/` (processed dataset + raw test split)

### 2) Train

Runs `accelerate launch ... main.py --train ...`.

```bash
bash scripts/train.sh
```

Outputs:

- `output_image_segmentation/final/` (saved at the end of training)
- intermediate checkpoints under `output_image_segmentation/` depending on `TrainingArguments`

### 3) Evaluate on the test split

```bash
bash scripts/evaluate.sh
```

Outputs:

- `output_evaluation/results_test.json`

### 4) Predict masks (and compute a custom mean IoU)

```bash
bash scripts/predict.sh
```

Outputs:

- `output_prediction/mask/*.bmp` (predicted masks)
- `output_prediction/result.json` (summary of mean IoU + loss)

## Test set results

The current evaluation output on the test split is recorded in `output_evaluation/results_test.json`:

- `mean_iou`: **0.9419186788822862**
- `mean_accuracy`: **0.9654617581129691**
- `overall_accuracy`: **0.9965993022340177**
- `test_loss`: **0.6843880406066553**
