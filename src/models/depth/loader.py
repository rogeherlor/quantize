"""
    VGGT, implemented in PyTorch.
    Original paper: 'VGGT: Visual Geometry Grounded Transformer' https://arxiv.org/abs/2503.11651.
"""

from src.models.depth.vggt.vggt.models.vggt import VGGT

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
    # net = VGGT.from_pretrained("facebook/VGGT-1B") if pretrained else VGGT
    net = VGGT.from_pretrained("../../../model_zoo/vggt/vggt_1B_commercial.pt") if pretrained else VGGT
    return net