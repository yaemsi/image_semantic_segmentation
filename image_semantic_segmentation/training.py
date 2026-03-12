import evaluate
import torch
import torch.nn as nn

import numpy as np

from transformers import (
    PreTrainedModel, 
    Trainer,
)

from typing import (
    Any,
    Dict,
    Tuple
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
    
    logits, labels = eval_pred
    metric = evaluate.load("mean_iou")

    # Upsample logits to match label size
    predictions = np.argmax(logits, axis=1)
    
    return metric.compute(
        predictions=predictions, 
        references=labels, 
        num_labels=2, 
        ignore_index=255
    )


class LogoSegmentationTrainer(Trainer):

    def compute_loss(
        self, 
        model: PreTrainedModel, 
        inputs: Dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None, 
        ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        
        labels = inputs.pop("labels")           # shape (batch, h, w)
        outputs = model(**inputs)               # Segformer outputs
        logits = outputs.logits                 # (batch, 2, h, w)

        # Option A: Keep CE (very stable baseline)
        # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=255)
        # loss = loss_fct(logits, labels.long())

        # Option B: Dice + BCE (your preference)
        preds = torch.softmax(logits, dim=1)[:, 1]   # prob of logo class
        preds = preds.unsqueeze(1)                   # (B,1,H,W)

        labels_binary = (labels == 1).float().unsqueeze(1)  # (B,1,H,W)

        # Reuse your earlier losses
        #dice_loss = DiceLoss()(preds, labels_binary)
        #dice_loss = nn.BCELoss()(preds, labels_binary)
        
        #bce_loss  = F.binary_cross_entropy_with_logits(logits[:,1,:,:], labels_binary.squeeze(1))
        # or F.binary_cross_entropy(preds, labels_binary)

        #loss = 0.5 * dice_loss + 0.5 * bce_loss
        loss = CombinedLoss()(preds, labels_binary)

        return (loss, outputs) if return_outputs else loss

