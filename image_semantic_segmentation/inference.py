

import glob
import os
import torch
import re
import json
import numpy as np

from loguru import logger
from PIL import Image

from datasets import (
    load_from_disk,
    load_dataset,
)

from torch.utils.data import DataLoader
from tqdm import tqdm


from image_semantic_segmentation import (
    gather_files,
    UNetPlusPlusConfig, 
    UNetPlusPlusHF,
    padding_fn,
    compute_metrics
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
    logger.info(f"pred_file_list == {pred_file_list}")
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
    ):

    logger.info(f"********* Running evaluation *********")

    
    logger.info(f">> Loading the model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusHF.from_pretrained(model_path)
    model.to(device)

    logger.info(f">> Loading the dataset ...")
    ds = load_from_disk(dataset_path)
    ds[f'{split}'].set_transform(padding_fn)
    
    dataloader = DataLoader(
        ds[f'{split}'], 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_proc
        )
    preds, labels = [], []


    logger.info(f">> Running inference ...")
    for data in tqdm(dataloader):
        inputs = {k: v.to(device) for k,v in data.items() if isinstance(v, torch.tensor)}
        annotations = inputs.pop("labels")
        outputs = model(**inputs)
        preds.extend(outputs.tolist())
        labels.extend(annotations.tolist())
    
    logger.info(f">> Computing metrics ...\n")
    results = compute_metrics(preds, labels)

    logger.info(f"{results}")

    logger.info(f"\n>> Saving results ...\n")
    with open(f"{os.path.join(save_path, 'results.json')}", "w") as f:
        json.dump(results, f, indent = 4)
    logger.info(f"**************** Done ****************")

    





def predict(
    model_path: str | os.PathLike, 
    img_path: str | os.PathLike, 
    save_path: str | os.PathLike,
    batch_size: int = 32,
    num_proc: int = 2
    ):
    logger.info(f"********* Running inference *********")

    logger.info(f">> Loading the model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusHF.from_pretrained(model_path)
    model.to(device)


    logger.info(f">> Building the dataset ...")
    
    # Gathering files names
    imgs = gather_files(f"{img_path}")
    names = [f.split('.')[0] for f in imgs]
    
    # Loading datasets 
    img_ds = load_dataset(path = f"{img_path}", data_files = imgs)
    
    # Adding name column & renaming label column
    img_ds['train'] = img_ds['train'].add_column('name', names)
    img_ds = img_ds.rename_column("image", "pixel_values")


    img_ds[f'train'].set_transform(padding_fn)
    
    dataloader = DataLoader(
        img_ds[f'train'], 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_proc
        )
    
    preds = []


    logger.info(f">> Running inference ...")
    for data in tqdm(dataloader):
        inputs = {k: v.to(device) for k,v in data.items() if isinstance(v, torch.tensor)}
        outputs = model(**inputs)
        preds.extend(outputs.tolist())
    
    logger.info(f">> Saving predictions to disk ...\n")
    for mask, name in zip(preds, names):


    logger.info(f"*************** Done ****************")


