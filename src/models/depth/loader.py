"""
    VGGT, implemented in PyTorch.
    Original paper: 'VGGT: Visual Geometry Grounded Transformer' https://arxiv.org/abs/2503.11651.
"""

import os
import torch
from src.models.depth.vggt.vggt.models.vggt import VGGT

def vggt(pretrained=False, **kwargs):
    """
    VGGT model from '' .
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    """
    if pretrained:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        model_path = os.path.join(project_root, "data3", "rogelio", "model_zoo", "vggt", "vggt_1B_commercial.pt")

        net = VGGT(**kwargs)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict)
    else:
        net = VGGT(**kwargs)
    return net