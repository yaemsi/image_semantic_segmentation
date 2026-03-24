

import glob
import os
import torch
import re
import json
import numpy as np

from loguru import logger
from PIL import Image

from datasets import (
    concatenate_datasets,
    load_from_disk,
    load_dataset,
)

from torch.utils.data import DataLoader
from transformers import EvalPrediction
from tqdm import tqdm


from image_semantic_segmentation import (
    IMAGE_H,
    PAD_H,
    IMAGE_W,
    PAD_W,
    gather_files,
    UNetPlusPlusHF,
    padding_fn,
    compute_metrics,
    simple_loss_func
)

def iou(pred: torch.Tensor, mask:torch.Tensor, label=1) -> float:
    """ A function that compute the Intersection over Union (IoU)
    for the pixels with a given label between the prediction and the mask
    """
    assert pred.shape == mask.shape
    pred_label = (pred == label).type(torch.int)
    mask_label = (mask == label).type(torch.int)

    intersection = pred_label * mask_label
    union = (pred_label + mask_label - intersection)

    iscore = intersection.sum().numpy()
    uscore = union.sum().numpy()

    assert uscore != 0, 'the label {} is not present in the pred and the mask'.format(label)

    return iscore / uscore


def custom_compute_metrics(metric: str, pred_path: str, mask_path: str) -> float:
    """
    A function that applies the metric to compare
    each image and mask located in the pred_path and mask_path respectively.
    """
    pred_file_list = sorted(glob.glob(pred_path + "/*"))
    score = 0.

    for pred_file in pred_file_list:

        filename = re.split('[/.]', pred_file)[-2]

        mask = torch.from_numpy(np.array(Image.open(os.path.join(mask_path, filename)+'.bmp'), dtype=np.uint8))/255
        pred = np.array(Image.open(pred_file), dtype=np.uint8)/255
        mask_pred = np.zeros_like(mask, dtype=np.uint8)
        mask_pred[pred > 0.5] = 1

        score += metric(torch.from_numpy(mask_pred), mask)
    return score/len(pred_file_list)



def evaluate(
    model_path: str | os.PathLike,
    dataset_path: str | os.PathLike,
    save_path: str | os.PathLike,
    split: str = 'test',
    batch_size: int = 32,
    num_proc: int = 2,
    ) -> None:

    logger.info(f"********* Running evaluation *********")


    logger.info(f">> Loading the model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusHF.from_pretrained(model_path)
    model.to(device)
    model.eval()

    logger.info(f">> Loading the dataset ...")
    ds = load_from_disk(dataset_path)
    ds[f'{split}'].set_transform(padding_fn)

    dataloader = DataLoader(
        ds[f'{split}'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_proc,
        pin_memory=True,
        prefetch_factor=num_proc,
        )
    preds, labels = [], []

    total_loss = 0.
    criterion = simple_loss_func
    logger.info(f">> Running inference ...")

    for data in tqdm(dataloader):
        inputs = {k: v.to(device) for k,v in data.items() if isinstance(v, torch.Tensor)}
        annotations = inputs.pop("labels")
        outputs = None
        with torch.no_grad():
            outputs = model(**inputs)
        loss = criterion(outputs, annotations) # "outputs" is of SemanticSegmenterOutput type

        total_loss += loss.item()
        preds.extend(outputs.logits.tolist())
        labels.extend(annotations.tolist())

    logger.info(f">> Computing metrics ...\n")
    results = compute_metrics(EvalPrediction(torch.tensor(preds), torch.tensor(labels)))
    results[f'{split}_loss'] = total_loss / len(dataloader)

    logger.info(f"{results}")

    logger.info(f"\n>> Saving results ...\n")
    os.makedirs(f"{save_path}", exist_ok=True)
    with open(f"{os.path.join(save_path, f'results_{split}.json')}", "w") as f:
        json.dump(results, f, indent = 4)
    logger.info(f"**************** Done ****************")





def predict(
    img_path: str | os.PathLike,
    save_path: str | os.PathLike,
    model_path: str | os.PathLike = "./output_image_segmentation/final",
    batch_size: int = 32,
    num_proc: int = 2
    ) -> None:
    logger.info(f"********* Running inference *********")

    logger.info(f">> Loading the model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusHF.from_pretrained(model_path)
    model.to(device)



    logger.info(f">> Building the dataset ...")

    # Gathering files names
    imgs = gather_files(os.path.join(f"{img_path}", "img"))
    masks = gather_files(os.path.join(f"{img_path}", "mask"))
    names = [f.split('.')[0] for f in imgs]


    # Loading datasets
    img_ds = load_dataset(path = os.path.join(f"{img_path}", "img"), data_files = imgs)
    msks_ds = load_dataset(path = os.path.join(f"{img_path}", "mask"), data_files = masks)

    # Adding name column & renaming label column
    img_ds['train'] = img_ds['train'].add_column('names', names)
    msks_ds = msks_ds.rename_column("image", "labels")
    img_ds = img_ds.rename_column("image", "pixel_values")

    # Merging into one dataset
    img_ds['train'] = concatenate_datasets([img_ds['train'], msks_ds['train']], axis=1)


    img_ds.set_transform(padding_fn)

    dataloader = DataLoader(
        img_ds['train'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_proc,
        pin_memory=True,
        prefetch_factor=num_proc,
        )


    logger.info(f">> Running inference ...")
    total_loss = 0.
    save_dir = os.path.join(save_path, 'mask')
    os.makedirs(save_dir, exist_ok = True)

    for data in tqdm(dataloader):
            inputs = {k: v.to(device) for k,v in data.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                outputs = model(**inputs)

            for pred, name in zip(outputs.logits, data['names']):
                mask = np.argmax(pred.cpu(), axis=0)                     # (2, H, W) -> (H, W)
                mask = mask[PAD_H:IMAGE_H+PAD_H,PAD_W:IMAGE_W+PAD_W]     # (H, W) -> (H', W')
                mask[mask==1] = 255
                pil_image = Image.fromarray(mask.numpy().astype(np.uint8), 'L')
                pil_image.save(os.path.join(f"{save_dir}", f"{name}.bmp"), "BMP")

            total_loss += outputs.loss.item()


    logger.info(f">> Running custom evaluation ...\n")
    score = custom_compute_metrics(iou, save_dir, os.path.join(img_path, 'mask'))
    logger.info('>> Mean IOU: {:.4f}'.format(score))
    res = {'Mean IOU': score, 'Loss': total_loss / len(dataloader)}
    with open(os.path.join(save_path, "result.json"), 'w') as fp:
        json.dump(res, fp, indent = 4)

    logger.info(f"*************** Done ****************")
