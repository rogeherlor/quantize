"""
Layer selection and filtering utilities for stats collection and distillation.
Shared between run_stats.py and run_distill.py.
"""

from src.logger import logger


def expand_layer_names(model, layer_specs):
    """
    Expand layer specifications to include all sub-layers.
    
    Args:
        model: PyTorch model
        layer_specs: List of layer specifications (e.g., ["aggregator.frame_blocks.12", "camera_head"])
    
    Returns:
        Set of fully expanded layer names including all sub-modules and parameters
    """
    if not layer_specs:
        # If no layer specs, include all layers
        return None
    
    expanded_names = set()
    
    for layer_spec in layer_specs:
        # Try to navigate to the module
        try:
            parts = layer_spec.split('.')
            current = model
            
            for part in parts:
                if '[' in part and ']' in part:
                    attr_name, idx = part.split('[')
                    idx = int(idx.rstrip(']'))
                    current = getattr(current, attr_name)[idx]
                else:
                    current = getattr(current, part)
            
            # Traverse all sub-modules under this module
            for name, _ in current.named_modules():
                if name == '':
                    # The module itself
                    expanded_names.add(layer_spec)
                else:
                    # Sub-modules
                    expanded_names.add(f"{layer_spec}.{name}")
            
            # Also add all parameters under this module
            for name, _ in current.named_parameters():
                param_full_name = f"{layer_spec}.{name}"
                expanded_names.add(param_full_name)
                
        except (AttributeError, IndexError) as e:
            logger.warning(f"Could not expand layer spec '{layer_spec}': {e}")
            # Still add the original spec in case it's a valid parameter name
            expanded_names.add(layer_spec)
    
    logger.info(f"Expanded {len(layer_specs)} layer specs to {len(expanded_names)} layer names")
    return expanded_names


def filter_stats_by_layers(stats_dict, layer_filter):
    """
    Filter a stats dictionary based on layer specifications.
    
    Args:
        stats_dict: Dict of stats {layer_name: stats_data}
        layer_filter: Set of layer names to include (None = include all)
    
    Returns:
        Filtered stats dict
    """
    if layer_filter is None:
        return stats_dict
    
    filtered = {}
    for key, value in stats_dict.items():
        # Extract layer name without suffix (.weight, .activation_out, .grad)
        layer_name = key.rsplit('.', 1)[0] if '.' in key else key
        
        # Check if this layer matches any in our filter
        # Use prefix matching to catch sub-layers
        for filter_name in layer_filter:
            if layer_name == filter_name or layer_name.startswith(filter_name + '.'):
                filtered[key] = value
                break
    
    logger.info(f"Filtered stats: {len(filtered)} / {len(stats_dict)} entries match layer filter")
    return filtered


def filter_snapshots_by_layers(snapshots, layer_filter):
    """
    Filter snapshots dictionary based on layer specifications.
    
    Args:
        snapshots: Dict of snapshots {layer_name: tensor}
        layer_filter: Set of layer names to include (None = include all)
    
    Returns:
        Filtered snapshots dict
    """
    return filter_stats_by_layers(snapshots, layer_filter)
