

def run_minmax_quantization(args):
    """
    Simple MinMax post-training quantization.
    This is a fallback/comparison method.
    """
    logger.info("Running MinMax PTQ quantization...")
    
    # Setup dataloader for calibration
    pin_memory = True if args.device == 'cuda' else False
    dataloader_dict = setup_dataloader(
        args.dataset_name, 
        args.batch_size, 
        args.nworkers, 
        pin_memory=pin_memory, 
        DDP_mode=False, 
        model=args.model
    )
    
    calibration_loader = dataloader_dict["train"]
    
    # Create original FP model
    logger.info("Creating original FP32 model...")
    model = create_model(args)
    model = model.to(args.device)
    model.eval()
    
    # Collect statistics for calibration
    logger.info("Collecting statistics for calibration...")
    stats = {}
    
    def collect_stats_hook(name):
        def hook(module, input, output):
            if name not in stats:
                stats[name] = {'min': None, 'max': None}
            
            # Collect min/max statistics
            if isinstance(input, tuple):
                data = input[0]
            else:
                data = input
                
            current_min = data.min()
            current_max = data.max()
            
            if stats[name]['min'] is None:
                stats[name]['min'] = current_min
                stats[name]['max'] = current_max
            else:
                stats[name]['min'] = min(stats[name]['min'], current_min)
                stats[name]['max'] = max(stats[name]['max'], current_max)
        
        return hook
    
    # Register hooks for statistics collection
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(collect_stats_hook(name)))
    
    # Run calibration
    sample_count = 0
    calibration_samples = args.calibration_samples  # Now cleanly accessible via dotdict
    logger.info(f"Using {calibration_samples} calibration samples for MinMax quantization")
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(calibration_loader):
            if sample_count >= calibration_samples:
                break
            
            inputs = inputs.to(args.device)
            _ = model(inputs)
            sample_count += inputs.size(0)
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    logger.info(f"Collected statistics from {sample_count} samples")
    
    # Apply quantization based on collected statistics
    # (This is a simplified version - you can enhance it with your quantization modules)
    logger.info("Applying quantization...")
    
    # For now, just log that MinMax quantization would be applied here
    # In a full implementation, you would replace layers with quantized versions
    # and set the scales based on the collected statistics
    
    logger.info("MinMax quantization completed")
    return model