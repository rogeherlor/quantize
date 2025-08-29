"""
    VGGT, implemented in PyTorch.
    Original paper: 'VGGT: Visual Geometry Grounded Transformer' https://arxiv.org/abs/2503.11651.
"""

import os
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
    if pretrained:
        # Get absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        model_path = os.path.join(project_root, "model_zoo", "vggt", "vggt_1B_commercial.pt")
        
        # Create model instance
        net = VGGT(**kwargs)
        
        # Load state dict from local .pt file
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict)
    else:
        net = VGGT(**kwargs)
    return net