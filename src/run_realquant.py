import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.logger import logger
import src.module_quantization as Q

from hydra import initialize, compose
from src.models.depth.vggt.training.trainer import Trainer

from src.run_distill import selective_quantize_layers
from src.models.depth.qvggt import run_evaluation_vggt

QLAYERS = [
        "aggregator.frame_blocks.6", "aggregator.frame_blocks.7", 
        "aggregator.frame_blocks.8", "aggregator.frame_blocks.9",
        "aggregator.frame_blocks.10", "aggregator.frame_blocks.11",
        "aggregator.global_blocks.6", "aggregator.global_blocks.7",
        "aggregator.global_blocks.8", "aggregator.global_blocks.9",
        "aggregator.global_blocks.10", "aggregator.global_blocks.11"
    ]

class W8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Store weights as INT8
        self.register_buffer('weight_int8', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('scale', torch.tensor(1.0))
        
        # Activation quantization parameters
        self.register_buffer('x_scale', torch.tensor(1.0))
        self.register_buffer('use_x_quant', torch.tensor(False))
        
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_features)))
        else:
            self.register_parameter('bias', None)

    @torch.no_grad()
    def load_from_qlinear(self, qlinear: Q.QLinear):
        # 1. Get scale and weights from QLinear
        # QExample: w_int = round(w_float / scale).clamp(...)
        scale = qlinear.w_Qparms['scale']
        if scale.numel() > 1:
            self.scale = scale.detach().clone()
        else:
            self.scale.copy_(scale.reshape(()))
        
        # Load activation scale if present
        if qlinear.x_quantizer is not None:
             # LSQ x_scale
             if 'scale' in qlinear.x_Qparms:
                 src_scale = qlinear.x_Qparms['scale']
                 if src_scale.numel() > 1:
                     self.x_scale = src_scale.detach().clone()
                 else:
                     self.x_scale.copy_(src_scale.reshape(()))
                 self.use_x_quant.fill_(True)
                 logger.info("Enabled W8A8 for layer")
        
        if qlinear.bias is not None:
            self.bias.copy_(qlinear.bias)

        # 2. Quantize and clamp to int8 range [-128, 127]
        # LSQ usually uses [-2^(b-1), 2^(b-1)-1]. For 8 bit: [-128, 127]
        w_float = qlinear.weight.data
        w_int = (w_float / scale).round().clamp(-128, 127).to(torch.int8)
        self.weight_int8.copy_(w_int)

    def forward(self, input):
        # REAL W8A8 Inference
        if self.use_x_quant and input.is_cuda and hasattr(torch, '_int_mm'):
            # 1. Quantize Activations to INT8
            # x_int = clamp(round(x / x_scale))
            x_int8 = (input / self.x_scale).round().clamp(-128, 127).to(torch.int8)
            
            # 2. Integer Matrix Multiplication (Real Acceleration)
            # torch._int_mm requires 2D inputs: (M, K) @ (N, K).t() -> (M, N) int32
            # Handle shapes
            input_shape = input.shape
            if input.dim() > 2:
                x_flat = x_int8.view(-1, input_shape[-1])
            else:
                x_flat = x_int8
            
            # Perform int8 matmul. Weight is [out, in], so we transpose in the call concept or use t()
            # _int_mm(mat1, mat2) -> mat1 @ mat2
            # We want: x @ w.T
            # w_int8 is [out, in]. w_int8.T is [in, out].
            # Shapes: [M, In] @ [In, Out] -> [M, Out]
            out_int32 = torch._int_mm(x_flat, self.weight_int8.t())
            
            # 3. Dequantize Output
            # y = y_int32 * w_scale * x_scale
            out_result = out_int32.to(input.dtype) * (self.scale * self.x_scale)
            
            # 4. Reshape and Add Bias
            if input.dim() > 2:
                out_result = out_result.view(*input_shape[:-1], self.out_features)
            
            if self.bias is not None:
                out_result += self.bias
                
            return out_result
            
        else:
            # Fallback to W8A16 if x_quant missing or CPU
            # On-the-fly Dequantization (W8A16/W8A32)
            w_dequant = self.weight_int8.to(input.dtype) * self.scale
            return F.linear(input, w_dequant, self.bias)

class W4Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Store weights as Packed INT4 (2 weights per UINT8)
        # Shape: [out_features, in_features // 2]
        self.register_buffer('weight_packed', torch.zeros((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer('scale', torch.tensor(1.0))
        
        # Activation quantization parameters
        self.register_buffer('x_scale', torch.tensor(1.0))
        self.register_buffer('use_x_quant', torch.tensor(False))
        
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_features)))
        else:
            self.register_parameter('bias', None)

    @torch.no_grad()
    def load_from_qlinear(self, qlinear: Q.QLinear):
        scale = qlinear.w_Qparms['scale']
        if scale.numel() > 1:
            self.scale = scale.detach().clone()
        else:
            self.scale.copy_(scale.reshape(()))
        
        # Load activation scale if present
        if qlinear.x_quantizer is not None:
             if 'scale' in qlinear.x_Qparms:
                 # Handle potentially per-channel scales by detaching/cloning
                 src_scale = qlinear.x_Qparms['scale']
                 if src_scale.numel() > 1:
                     self.x_scale = src_scale.detach().clone()
                 else:
                     self.x_scale.copy_(src_scale.reshape(()))
                 self.use_x_quant.fill_(True)
                 logger.info("Enabled W4A4 for layer")
                 
        if qlinear.bias is not None:
            self.bias.copy_(qlinear.bias)

        # 1. Quantize to signed int4 range [-8, 7]
        w_float = qlinear.weight.data
        w_int = (w_float / scale).round().clamp(-8, 7).to(torch.int8)
        
        # 2. Convert to unsigned 0-15 by adding 8
        w_uint = (w_int + 8).to(torch.uint8) 
        
        # 3. Pack two columns into one byte
        # Even indices: low 4 bits, Odd indices: high 4 bits
        # Note: This requires in_features to be even.
        if w_uint.shape[1] % 2 != 0:
            # Padding not handled in this simple example
            logger.warning(f"Feature dim {w_uint.shape[1]} not divisible by 2. Packing skipped.")
            return
            
        w_low = w_uint[:, 0::2]
        w_high = w_uint[:, 1::2]
        self.weight_packed.copy_((w_high << 4) | w_low)

    def forward(self, input):
        # On-the-fly Dequantization (W4A16 or W4A4)
        
        if self.use_x_quant and input.is_cuda and hasattr(torch, '_int_mm'):
             # W4A4 (executed as W4->W8A8 logic)
             # This provides acceleration by using INT8 Tensor Cores
             
             # 1. Quantize Activations to INT8 (clamped to 4-bit range)
             # Range for 4-bit signed is [-8, 7]
             x_int8 = (input / self.x_scale).round().clamp(-8, 7).to(torch.int8)

             # 2. Unpack Weights (W4 -> W8)
             # We unpack to int8 to use _int_mm
             w_packed = self.weight_packed
             w_low = (w_packed & 0x0F)
             w_high = (w_packed >> 4)
             
             # Interleave to restore original shape [Out, In]
             w_unpacked_uint8 = torch.stack([w_low, w_high], dim=2).flatten(1)
             w_int8 = (w_unpacked_uint8.to(torch.int8) - 8) # Restore sign: 0..15 -> -8..7
             
             # 3. INT8 Matmul
             # Flatten input if needed
             input_shape = input.shape
             if input.dim() > 2:
                x_flat = x_int8.view(-1, input_shape[-1])
             else:
                x_flat = x_int8

             # MatMul: [M, In] @ [In, Out] -> [M, Out]
             # w_int8 is [Out, In], so we transpose
             out_int32 = torch._int_mm(x_flat, w_int8.t())
             
             # 4. Dequantize
             out_result = out_int32.to(input.dtype) * (self.scale * self.x_scale)
             
             # 5. Reshape and Bias
             if input.dim() > 2:
                out_result = out_result.view(*input_shape[:-1], self.out_features)
            
             if self.bias is not None:
                out_result += self.bias
            
             return out_result

        else:
            # W4A16 Fallback
            # 1. Unpack
            w_packed = self.weight_packed
            
            # Low 4 bits
            w_low = (w_packed & 0x0F)
            # High 4 bits
            w_high = (w_packed >> 4)
            
            # Interleave to restore original shape
            w_unpacked = torch.stack([w_low, w_high], dim=2).flatten(1)
            
            # 2. Restore sign (subtract 8) and scale
            w_dequant = (w_unpacked.to(input.dtype) - 8.0) * self.scale
            
            return F.linear(input, w_dequant, self.bias)

def pack_model_to_real_int(model, layer_names):
    """
    Replaces QLinear layers with W8Linear or W4Linear based on their configuration.
    """
    logger.info("Packing model to REAL integer weights...")
    
    # We need to iterate carefully to replace modules in-place
    # 1. Collect modules to replace to avoid modifying dict while iterating
    modules_to_replace = {}
    
    for name, module in model.named_modules():
        # Check if name starts with any of the layer_names (to handle blocks)
        is_target = False
        for target in layer_names:
            if name == target or name.startswith(target + "."):
                is_target = True
                break

        if is_target and isinstance(module, Q.QLinear):
            if module.num_bits == 8:
                new_module = W8Linear(module.in_features, module.out_features, module.bias is not None)
            elif module.num_bits == 4:
                new_module = W4Linear(module.in_features, module.out_features, module.bias is not None)
            else:
                logger.warning(f"Layer {name} has {module.num_bits} bits. Only 4 and 8 are supported for packing. Skipping.")
                continue
            
            # Transfer device
            new_module.to(module.weight.device)
            new_module.load_from_qlinear(module)
            modules_to_replace[name] = new_module
            
    # 2. Replace
    for name, new_module in modules_to_replace.items():
        # Handle hierarchical naming replacement
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                 parent = parent[int(part)]
            else:
                 parent = getattr(parent, part)
        
        final_part = parts[-1]
        
        if final_part.isdigit():
             parent[int(final_part)] = new_module
        else:
             setattr(parent, final_part, new_module)

        dtype_str = str(new_module.weight_int8.dtype) if hasattr(new_module, 'weight_int8') else 'Packed4'
        logger.info(f"Replaced {name} with Real Int {dtype_str} Layer")
        
    return model

def get_model_size_mb(model):
    total_bytes = 0
    seen_params = set()
    
    # buffers incudes registered integer tensors
    for param in list(model.parameters()) + list(model.buffers()):
        if param in seen_params:
            continue
        seen_params.add(param)
        
        total_bytes += param.numel() * param.element_size()
            
    return total_bytes / 1024 / 1024


def run_realquant_vggt(rank, args):

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12399"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    with initialize(version_base=None, config_path="models/depth/vggt/training/config"):
        cfg = compose(config_name="default")
    
    trainer = Trainer(**cfg)
    original_model = trainer.model.module

    quant_model = selective_quantize_layers(original_model, args, QLAYERS)

    logger.info(f"Loading checkpoint from {args.init_from}")
    checkpoint = torch.load(args.init_from, map_location="cpu")
    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    else:
        model_state_dict = checkpoint
    missing, unexpected = trainer.model.module.load_state_dict(
        model_state_dict, strict=False
    )
    logger.info(f"Model loaded. Missing keys: {len(missing) if missing else 0}, Unexpected keys: {len(unexpected) if unexpected else 0}")
    
    quant_model_real = pack_model_to_real_int(quant_model, QLAYERS)

    logger.info(f"Real Quantized model size (Packed): {get_model_size_mb(quant_model):.2f} MB")
    print(quant_model_real)
    quant_model_real.eval()
    start_time = time.time()
    run_evaluation_vggt(quant_model_real)
    end_time = time.time()
    logger.info(f"Real Quantized model evaluation time: {end_time - start_time:.2f} seconds")
    
    logger.info(f"Fake Quantized model size: {get_model_size_mb(quant_model):.2f} MB")
    quant_model.eval()
    start_time = time.time()
    run_evaluation_vggt(quant_model)
    end_time = time.time()
    logger.info(f"Fake Quantized model evaluation time: {end_time - start_time:.2f} seconds")
    
    logger.info(f"Original model size: {get_model_size_mb(original_model):.2f} MB")
    original_model.eval()
    start_time = time.time()
    run_evaluation_vggt(original_model)
    end_time = time.time()
    logger.info(f"Original model evaluation time: {end_time - start_time:.2f} seconds")

# TODO: benchmarking and profiling. Watch kernels and bottlenecks.
def run_realquant(args):
    if args.ddp == True:
        logger.error("DDP for REALQUANT is not supported.")
    else:
        run_realquant_vggt(0, args)