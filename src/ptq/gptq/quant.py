import torch
import torch.nn as nn

def quantize(x, scale, zero_point, maxq):
    """
    Simple quantization function compatible with GPTQ.
    
    Args:
        x: Input tensor to quantize
        scale: Quantization scale
        zero_point: Zero point (typically 0 for symmetric quantization)
        maxq: Maximum quantization value (2^bits - 1)
    
    Returns:
        Quantized tensor
    """
    if scale.numel() == 1:
        scale = scale.item()
    
    if isinstance(zero_point, torch.Tensor) and zero_point.numel() == 1:
        zero_point = zero_point.item()
    
    # Quantize
    q = x / scale + zero_point
    q = torch.clamp(torch.round(q), 0, maxq)
    
    # Dequantize 
    return (q - zero_point) * scale
