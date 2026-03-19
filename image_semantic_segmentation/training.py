import evaluate
import torch
import torch.nn as nn

import numpy as np

from loguru import logger
from transformers import (
    PreTrainedModel, 
    Trainer,
)

from transformers.modeling_outputs import (
    SemanticSegmenterOutput,
)

from typing import (
    Any,
    Dict,
    Tuple
)

from image_semantic_segmentation import (
    IMAGE_H, 
    PAD_H,
    IMAGE_W, 
    PAD_W,
)

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        # preds: [batch, 1, H, W] - sigmoid outputs (0 to 1)
        # targets: [batch, 1, H, W] - binary masks (0 or 1)
        
        # Flatten to 1D for easier computation
        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (preds_flat * targets_flat).sum(dim=1)
        union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()  # Average over batch


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, smooth: float = 1e-6) -> None:
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Weight for Dice (1 - alpha for BCE)
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        dice_loss = self.dice(preds, targets)
        bce_loss = self.bce(preds, targets)
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss


def compute_metrics(
    eval_pred: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, Any]:
    
    logits, labels = eval_pred # (B, 2, H, W) x (B, H, W)
    

    metric = evaluate.load("mean_iou")

    # Upsample logits to match label size
    predictions = np.argmax(logits, axis=1)

    # Manual removal of padding
    predictions = predictions[:,PAD_H:IMAGE_H-PAD_H,PAD_W:IMAGE_W-PAD_W]   # (B, H',W)
    labels = labels[:,PAD_H:IMAGE_H-PAD_H,PAD_W:IMAGE_W-PAD_W]             # (B, H',W)
    
    res = metric.compute(
        predictions=predictions, 
        references=labels, 
        num_labels=2, 
        ignore_index=255
    )
    for k in res.keys():
        v = res[k]
        if isinstance(v, np.ndarray):
            res[k] = v.tolist()
    return res



def custom_loss_func(
                outputs: SemanticSegmenterOutput,
                labels: torch.Tensor,
                num_items_in_batch = None,
            ):
    logits = outputs.logits     # (batch, 2, h, w)

    # Option B: Dice + BCE (your preference)
    preds = torch.softmax(logits, dim=1)[:, 1]         # prob of logo class
    preds = preds.unsqueeze(1)                         # (B,1,H,W)
    labels_binary = (labels == 1).float().unsqueeze(1) # (B,1,H,W)


    # Manual removal of padding
    preds = preds[:,PAD_H:IMAGE_H-PAD_H,PAD_W:IMAGE_W-PAD_W]                     # (B,1,H',W)
    labels_binary = labels_binary[:,PAD_H:IMAGE_H-PAD_H,PAD_W:IMAGE_W-PAD_W]     # (B,1,H',W)

    loss = nn.BCELoss()(preds, labels_binary)

    return loss

