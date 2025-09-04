import torch
import torch.nn as nn
import time
from functools import partial

from src.logger import logger
from src.utils import setup_dataloader, replace_module
from src.models.model import create_model
from src.module_quantization import QLinear, QConv2d
from src.run_qat import run_load_model

from .gptq import GPTQ

def find_layers(module, layers=None):
    """Find all Linear and Conv2d layers in a module."""
    if layers is None:
        layers = [nn.Linear, nn.Conv2d]
    
    if type(module) in layers:
        return {None: module}
    
    res = {}
    for name, child in module.named_children():
        res.update({name + '.' + k if k else name: v for k, v in find_layers(child, layers).items()})
    
    return res


def quantize_layer_with_gptq(layer, dataloader, args, num_samples=None):
    """
    Quantize a single layer using GPTQ algorithm integrated with your quantization framework.
    """
    if num_samples is None:
        num_samples = args.calibration_samples  # Clean dotdict access
    
    logger.info(f"Quantizing layer: {layer.__class__.__name__} with {num_samples} samples")
    
    # Create GPTQ instance for this layer
    gptq = GPTQ(layer)
    
    # Create a custom quantizer that bridges GPTQ with your framework
    class CustomQuantizer:
        def __init__(self, args):
            self.num_bits = args.num_bits
            self.scale = None
            self.zero = None
            self.maxq = 2 ** args.num_bits - 1
            
        def ready(self):
            return self.scale is not None
            
        def find_params(self, x, weight=True):
            """Initialize quantization parameters using your framework's logic."""
            # Create temporary quantization parameters
            w_Qparms = {'scale': nn.Parameter(torch.tensor([1.0], device=x.device))}
            
            if isinstance(layer, nn.Linear):
                w_Qn = -(2**(self.num_bits-1))
                w_Qp = 2**(self.num_bits-1) - 1
            elif isinstance(layer, nn.Conv2d):
                w_Qn = -(2**(self.num_bits-1))
                w_Qp = 2**(self.num_bits-1) - 1
            else:
                w_Qn = -(2**(self.num_bits-1))
                w_Qp = 2**(self.num_bits-1) - 1
            
            # Use your initializer if available
            if args.w_initializer:
                try:
                    args.w_initializer(weight, w_Qparms, w_Qn, w_Qp, None)
                    self.scale = w_Qparms['init_scale'].clone().detach()
                except:
                    # Fallback to simple initialization
                    self.scale = 2 * x.abs().mean() / (2**(self.num_bits-1) - 1)
            else:
                # Simple min-max initialization
                self.scale = 2 * x.abs().mean() / (2**(self.num_bits-1) - 1)
            
            self.zero = 0  # Symmetric quantization
            
            # Ensure scale is positive and reasonable
            if self.scale <= 0:
                self.scale = torch.tensor(0.01, device=x.device)
            
            logger.debug(f"Initialized quantizer with scale: {self.scale}")
    
    # Set up the custom quantizer
    gptq.quantizer = CustomQuantizer(args)
    
    # Collect statistics by running forward passes
    layer.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
                
            inputs = inputs.to(args.device)
            
            # Hook to capture input and output
            def hook_fn(module, input, output):
                gptq.add_batch(input[0], output)
            
            handle = layer.register_forward_hook(hook_fn)
            
            try:
                # Forward pass to collect statistics
                _ = layer(inputs)
                sample_count += inputs.size(0)
            finally:
                handle.remove()
    
    logger.info(f"Collected statistics from {sample_count} samples")
    
    # Run GPTQ quantization with clean dotdict access
    gptq.fasterquant(
        blocksize=args.gptq_blocksize,
        percdamp=args.gptq_percdamp,
        groupsize=args.gptq_groupsize,
        actorder=args.gptq_actorder,
        static_groups=args.gptq_static_groups
    )
    
    # Clean up
    gptq.free()
    
    logger.info("Layer quantization completed")


def quantize_model_gptq(model, dataloader, args):
    """
    Quantize an entire model using GPTQ algorithm.
    
    Args:
        model: The model to quantize
        dataloader: DataLoader for calibration data
        args: Configuration arguments
    
    Returns:
        Quantized model
    """
    logger.info("Starting GPTQ model quantization...")
    
    # Find all quantizable layers
    layers_dict = find_layers(model)
    logger.info(f"Found {len(layers_dict)} layers to quantize")
    
    # Process layers sequentially to manage memory
    for layer_name, layer in layers_dict.items():
        if layer_name:
            logger.info(f"Processing layer: {layer_name}")
        
        # Quantize this layer
        quantize_layer_with_gptq(layer, dataloader, args)
    
    logger.info("GPTQ model quantization completed")
    return model


def replace_with_quantized_modules(model, args):
    """
    Replace standard modules with quantized versions after GPTQ.
    This creates the final quantized model using your QLinear/QConv2d modules.
    """
    logger.info("Replacing modules with quantized versions...")
    
    replacement_dict = {
        nn.Conv2d: partial(QConv2d,
                          num_bits=args.num_bits,
                          w_grad_scale_mode=getattr(args, 'w_grad_scale_mode', 'LSQ_grad_scale'),
                          x_grad_scale_mode=getattr(args, 'x_grad_scale_mode', 'LSQ_grad_scale'),
                          weight_norm=getattr(args, 'weight_norm', None),
                          w_quantizer=args.w_quantizer,
                          x_quantizer=args.x_quantizer,
                          w_initializer=args.w_initializer,
                          x_initializer=args.x_initializer),
        nn.Linear: partial(QLinear,
                          num_bits=args.num_bits,
                          w_grad_scale_mode=getattr(args, 'w_grad_scale_mode', 'LSQ_grad_scale'),
                          x_grad_scale_mode=getattr(args, 'x_grad_scale_mode', 'LSQ_grad_scale'),
                          weight_norm=getattr(args, 'weight_norm', None),
                          w_quantizer=args.w_quantizer,
                          x_quantizer=args.x_quantizer,
                          w_initializer=args.w_initializer,
                          x_initializer=args.x_initializer)
    }
    
    exception_dict = {
        '__first__': partial(QConv2d, 
                            num_bits=getattr(args, 'first_bits', 8),
                            w_quantizer=getattr(args, 'w_first_last_quantizer', None),
                            x_quantizer=getattr(args, 'x_first_last_quantizer', None),
                            first_layer=True),
        '__last__': partial(QLinear,
                           num_bits=getattr(args, 'last_bits', 8),
                           w_quantizer=getattr(args, 'w_first_last_quantizer', None),
                           x_quantizer=getattr(args, 'x_first_last_quantizer', None),
                           first_layer=False)
    }
    
    model = replace_module(model, 
                          replacement_dict=replacement_dict, 
                          exception_dict=exception_dict, 
                          arch=args.model)
    
    logger.info("Module replacement completed")
    return model


def run_gptq_quantization(args):
    logger.info("Starting GPTQ quantization pipeline...")
    
    pin_memory = True if args.device == 'cuda' else False
    dataloader_dict = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory=pin_memory, DDP_mode=False, model=args.model)

    calibration_loader = dataloader_dict["train"]
    
    logger.info("Creating original model...")
    original_net = create_model(args)
    original_net = original_net.to(args.device)
    original_net.eval()

    logger.info("Creating quantized model structure...")
    quantized_net = run_load_model(args)
    quantized_net.eval()

    # Run GPTQ quantization on the original model
    qmodel = quantize_model_gptq(original_net, calibration_loader, args)
    
    # Replace with your quantized modules for final deployment
    # (Optional: this creates a model compatible with your QAT framework)
    if args.replace_with_qmodules:
        qmodel = replace_with_quantized_modules(qmodel, args)
    
    return qmodel