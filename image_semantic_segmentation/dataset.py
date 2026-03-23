
import cv2
import os
import torch


import torch
import zipfile


import albumentations as A
import numpy as np

from albumentations.pytorch import ToTensorV2


from datasets import (
    load_dataset,
    concatenate_datasets
)
from os.path import isfile, join

from tqdm import tqdm

from typing import (
    List,
    Dict
)

# Save padding values
IMAGE_H, PAD_H = 340, 6
IMAGE_W, PAD_W = 512, 0



def extract_data(root_dir: str | os.PathLike, output_dir: str | os.PathLike) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(root_dir, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

def gather_files(root_dir: str | os.PathLike) -> List[str]:
    return sorted([f for f in os.listdir(root_dir) if isfile(join(root_dir, f))])

def build_dataset(
    files_root_dir: str |  os.PathLike, 
    ds_save_dir: str | os.PathLike,
    val_size: float = 0.05,
    test_size: float = 0.10,
    shuffle: bool = True,
    ) -> None:
    # Gathering files names
    imgs = gather_files(join(files_root_dir, "img"))
    masks = gather_files(join(files_root_dir, "mask"))
    names = [f.split('.')[0] for f in imgs]
    
    # Loading datasets 
    img_ds = load_dataset(path = join(files_root_dir, "img"), data_files = imgs)
    msks_ds = load_dataset(path = join(files_root_dir, "mask"), data_files = masks)
    
    # Adding name column & renaming label column
    img_ds['train'] = img_ds['train'].add_column('names', names)
    msks_ds = msks_ds.rename_column("image", "labels")
    img_ds = img_ds.rename_column("image", "pixel_values")

    # Merging into one dataset 
    img_ds['train'] = concatenate_datasets([img_ds['train'], msks_ds['train']], axis=1)

    # Splitting to train, dev and test sets 
    img_ds = img_ds['train'].train_test_split(test_size=val_size, shuffle=shuffle)
    val_ds = img_ds.pop("test")
    img_ds = img_ds['train'].train_test_split(test_size=test_size, shuffle=shuffle)
    img_ds["validation"] = val_ds

    # Saving the test dataset in raw format
    img_dir = os.path.join(ds_save_dir, "raw", "test")
    os.makedirs(f"{img_dir}", exist_ok=True)
    for example in tqdm(img_ds['test'], desc="Saving images"):
        image = example['pixel_values']
        mask = example['labels']
        name = example['names']
        # Generate unique filenames
        img_filename = f"{os.path.join(img_dir, 'img', name)}.jpg"
        msk_filename = f"{os.path.join(img_dir, 'mask', name)}.bmp"
        
        # The image column usually holds PIL objects after loading
        # Save the PIL image
        image.save(f"{img_filename}")
        mask.save(f"{msk_filename}", "BMP")

    # Saving to disk
    ds_dir = os.path.join(ds_save_dir, "processed")
    os.makedirs(f"{ds_dir}", exist_ok=True)
    img_ds.save_to_disk(f"{ds_dir}")


def seg_data_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    return {"pixel_values": pixel_values, "labels": labels}


def padding_fn(
    examples: Dict[str, torch.Tensor], 
    ) -> Dict[str, torch.Tensor]:
    transform_func = A.Compose([
    # Padding
    A.PadIfNeeded(
        min_height=None, 
        min_width=None, 
        pad_height_divisor=32, 
        pad_width_divisor=32, 
        border_mode=cv2.BORDER_CONSTANT, 
        fill=255,
        fill_mask=255
    ),
    ToTensorV2(),
    ])
    images = [np.array(image.convert("RGB")) for image in examples["pixel_values"]]

    # Ensure masks are single-channel (L) for segmentation
    masks = [np.array(mask.convert("L")) for mask in examples["labels"]]
    
    # Scale masks to 0,1
    masks = [(mask - mask.min()) / (mask.max() - mask.min()) for mask in masks]
    
    inputs = {"pixel_values": [], "labels": [], "names": []}

    for img, mask, name in zip(images, masks, examples["names"]):
        
        # Apply Padding
        padded = transform_func(image=img, mask=mask)
        
        inputs["pixel_values"].append(padded["image"].float())
        
        # Ensure mask is long type and scaled (0 and 1)
        inputs["labels"].append(padded["mask"].long())

        # Keeping other data
        inputs["names"].append(name)

    return inputs

def train_preprocess_fn(
    examples: Dict[str, torch.Tensor], 
    ) -> Dict[str, torch.Tensor]:
    
    # Define the transformations
    transform_func = A.Compose([
        # 1. Geometric: Handles both image and mask
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.0625, 0.0625), # Roughly maps to shift_limit
            scale=(-0.9, 1.1),              # 1 - scale_limit to 1 + scale_limit
            rotate=(-15, 15),              # rotate_limit
            p=0.5
        ),
        
        # 2. Photometric: Only affects the image
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.RandomBrightnessContrast(p=0.3),
        
        # 3. Robustness: CoarseDropout to force feature learning
        A.CoarseDropout(
            num_holes_range=(2, 4), 
            hole_height_range=(10, 20), 
            hole_width_range=(10, 20), 
            #num_holes_range=(3, 6),
            #hole_height_range=(10, 20),
            #hole_width_range=(10, 20),
            p=0.3),
        
        # 4. Normalization (using ImageNet stats)
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        # Padding
        A.PadIfNeeded(
            min_height=None, 
            min_width=None, 
            pad_height_divisor=32, 
            pad_width_divisor=32, 
            border_mode=cv2.BORDER_CONSTANT, 
            fill=255,
            fill_mask=255
        ),
        ToTensorV2(),
    ])

    images = [np.array(image.convert("RGB")) for image in examples["pixel_values"]]
    # Ensure masks are single-channel (L) for segmentation
    masks = [np.array(mask.convert("L")) for mask in examples["labels"]]

    # Scale masks to 0,1
    masks = [(mask - mask.min()) / (mask.max() - mask.min()) for mask in masks]
    
    inputs = {"pixel_values": [], "labels": []}
    for img, mask in zip(images, masks):
        # Apply Albumentations
        augmented = transform_func(image=img, mask=mask)
        inputs["pixel_values"].append(augmented["image"].float())

        # Ensure mask is long type and scaled (0 and 1)
        inputs["labels"].append(augmented["mask"].long())
    
    # Retaining original data
    #inputs["pixel_values"].extend(examples["pixel_values"])
    #inputs["labels"].extend(examples["labels"])

    return inputs


def eval_preprocess_fn(
    examples: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
    
    # Define the transformations
    transform_func = A.Compose([
        # Padding
        A.PadIfNeeded(
            min_height=None, 
            min_width=None, 
            pad_height_divisor=32, 
            pad_width_divisor=32, 
            border_mode=cv2.BORDER_CONSTANT, 
            fill=0,
            fill_mask=255
        ),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    images = [np.array(image.convert("RGB")) for image in examples["pixel_values"]]
    
    # Ensure masks are single-channel (L) for segmentation
    masks = [np.array(mask.convert("L")) for mask in examples["labels"]]

    # Scale masks to 0,1
    masks = [(mask - mask.min()) / (mask.max() - mask.min()) for mask in masks]
    
    inputs = {"pixel_values": [], "labels": []}
    
    for img, mask in zip(images, masks):
        # Apply Albumentations
        tensorized = transform_func(image=img, mask=mask)
        
        inputs["pixel_values"].append(tensorized["image"].float())
        # Ensure mask is long type and scaled (0 and 1)
        inputs["labels"].append(tensorized["mask"].long())
        
    return inputs
