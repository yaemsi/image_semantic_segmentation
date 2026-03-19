import torch
import segmentation_models_pytorch as smp

from argparse import Namespace
from loguru import logger
from transformers import (
    PretrainedConfig,
    PreTrainedModel
)
from transformers.modeling_outputs import (
    SemanticSegmenterOutput,
)
from typing import (
    Any,
    Dict,
    List,
)
from image_semantic_segmentation.training import (
    DiceLoss,
    CombinedLoss
)

class UNetPlusPlusConfig(PretrainedConfig):
    model_type = "unetplusplus"
    def __init__(
        self, 
        encoder_name: int = 'resnet34', 
        encoder_depth: int = 5, 
        encoder_weights: str = 'imagenet', 
        decoder_use_norm: str = 'batchnorm', 
        decoder_channels: List[int] = [256, 128, 64, 32, 16], 
        decoder_attention_type: str | None = None, 
        decoder_interpolation: str ='nearest', 
        in_channels: int = 3, 
        classes: int = 2, 
        activation: str | None = None, 
        aux_params: Any | None = None,
        **kwargs: Dict[str, Any]
        ) -> None:
        
        super().__init__(**kwargs)
        self.encoder_name = encoder_name
        self.encoder_depth = encoder_depth
        self.encoder_weights = encoder_weights
        self.decoder_use_norm = decoder_use_norm
        self.decoder_channels = decoder_channels
        self.decoder_attention_type = decoder_attention_type
        self.decoder_interpolation = decoder_interpolation
        self.in_channels = in_channels
        self.classes = classes
        self.activation = activation
        self.aux_params = aux_params

class UNetPlusPlusHF(PreTrainedModel):
    config_class = UNetPlusPlusConfig

    def __init__(self, config: Namespace) -> None:
        super().__init__(config)
        self.model = smp.UnetPlusPlus(
            encoder_name=config.encoder_name,
            encoder_depth=config.encoder_depth, 
            encoder_weights=config.encoder_weights, 
            decoder_use_norm=config.decoder_use_norm, 
            decoder_channels=tuple(config.decoder_channels), 
            decoder_attention_type=config.decoder_attention_type, 
            decoder_interpolation=config.decoder_interpolation, 
            in_channels=config.in_channels, 
            classes=config.classes, 
            activation=config.activation, 
            aux_params=config.aux_params
        )
    
    def forward(
        self, 
        pixel_values: torch.Tensor, 
        labels: torch.Tensor = None
        ) -> SemanticSegmenterOutput:
        logits = self.model(pixel_values)
        loss = None
        if labels is not None:
            # We reuse the DiceBCELoss logic from before
            loss_fct = CombinedLoss()
            # SMP outputs [Batch, Classes, H, W]
            # We take the 'logo' channel (index 1) for binary comparison
            loss = loss_fct(logits[:, 1, :, :], labels)
            loss = loss_fct(logits[:, 1, :, :], labels)
        if loss is not None:
            return SemanticSegmenterOutput(loss = loss, logits = logits)  
        else:
            return SemanticSegmenterOutput(logits = logits)