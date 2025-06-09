"""
    ViT Base 16 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale' https://arxiv.org/abs/2010.11929.
"""

__all__ = ['vitb16']

import os
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def get_vit(model_name=False, pretrained=False, **kwargs):
    pass

def vitb16(pretrained=False, **kwargs):
    """
    ViT B 16 model from '' .
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    """
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    net = vit_b_16(weights=weights, progress=True)
    return net