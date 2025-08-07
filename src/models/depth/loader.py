"""
    VGGT, implemented in PyTorch.
    Original paper: 'VGGT: Visual Geometry Grounded Transformer' https://arxiv.org/abs/2503.11651.
"""

__all__ = ['vggt']

import os, sys
path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path)
sys.path.append(os.path.join(path, "vggt"))

from vggt.vggt.models.vggt import VGGT

def get_vggt(model_name=False, pretrained=False, **kwargs):
    pass

def vggt(pretrained=False, **kwargs):
    """
    VGGT model from '' .
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    """
    net = VGGT.from_pretrained("facebook/VGGT-1B") if pretrained else VGGT
    return net