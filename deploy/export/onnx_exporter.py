"""
ONNX export for LSQ-quantized VGGT models.

Export path:
  QAT checkpoint (QLinear with learned scales)
      → ONNX opset 17 with QuantizeLinear / DequantizeLinear nodes
      → trt_builder.py → TensorRT engine

Usage:
  python -m deploy.export.onnx_exporter \
      --checkpoint model_zoo/pytorchcv/vggt_w8a8_lsq.pth \
      --output     deploy/engines/vggt_w8a8.onnx \
      --num_bits   8 \
      --x_quantizer LSQ_quantizer \
      --w_quantizer LSQ_quantizer \
      --num_frames  4
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import onnx
import onnxsim

import src.module_quantization as Q
from src.run_realquant import QLAYERS, pack_model_to_real_int
from src.run_distill import selective_quantize_layers
from src.models.depth.vggt.training.trainer import Trainer
from hydra import initialize, compose


# ──────────────────────────────────────────────────────────────────────────────
# Export-mode wrapper for QLinear
# ──────────────────────────────────────────────────────────────────────────────

class _ExportQLinear(nn.Module):
    """
    Drop-in replacement for QLinear at ONNX export time.

    Emits explicit QuantizeLinear + DequantizeLinear (Q/DQ) ops that TensorRT
    can fuse into INT8 tensor-core operations.

    For quantizers that used smooth or Hadamard transformations, the folded
    weight (already computed by _forward_common's eval-time cache) is passed in
    as `weight_data`. This means the rotation / smooth factor is baked into the
    ONNX constant weight tensor — zero runtime overhead.
    """

    def __init__(self, qlinear: Q.QLinear, folded_weight: torch.Tensor):
        super().__init__()
        in_f = qlinear.in_features
        out_f = qlinear.out_features

        # Frozen folded weight (may be rotated / smoothed)
        self.register_buffer('weight', folded_weight.detach().cpu().float())

        # Weight quantization parameters
        w_scale = qlinear.w_Qparms.get('scale')
        if w_scale is None:
            raise ValueError(f"No weight scale found in QLinear {type(qlinear)}")
        self.register_buffer('w_scale', w_scale.detach().cpu().float())
        self.w_Qn = int(qlinear.w_Qn)
        self.w_Qp = int(qlinear.w_Qp)

        # Activation quantization parameters (optional → W8A16 if absent)
        x_scale = qlinear.x_Qparms.get('scale')
        self.has_x_quant = x_scale is not None
        if self.has_x_quant:
            self.register_buffer('x_scale', x_scale.detach().cpu().float())
            self.x_Qn = int(qlinear.x_Qn)
            self.x_Qp = int(qlinear.x_Qp)

        # Bias
        if qlinear.bias is not None:
            self.register_buffer('bias', qlinear.bias.detach().cpu().float())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weight (fake-quant → ONNX sees Q/DQ pair)
        # Per-tensor symmetric: zero_point = 0
        w = torch.fake_quantize_per_tensor_affine(
            self.weight.to(x.dtype),
            self.w_scale.item(),
            0,
            -self.w_Qn,
            self.w_Qp,
        )

        # Quantize activation if W8A8 (or W4A4)
        if self.has_x_quant:
            x = torch.fake_quantize_per_tensor_affine(
                x,
                self.x_scale.item(),
                0,
                -self.x_Qn,
                self.x_Qp,
            )

        return nn.functional.linear(x, w, self.bias)


def _replace_qlinear_with_export(model: nn.Module, layer_prefixes: list) -> nn.Module:
    """
    Replaces every QLinear in `layer_prefixes` with _ExportQLinear.

    Runs a warm-up forward (no-op dummy) first to ensure Hadamard / smooth
    caches (`module.rotated_weight`) are populated — the folded weight is then
    extracted directly from that cache.
    """
    replacements = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, Q.QLinear):
            continue
        is_target = any(name == p or name.startswith(p + '.') for p in layer_prefixes)
        if not is_target:
            continue

        # Use cached rotated/smoothed weight if available; fall back to raw weight
        folded = mod.rotated_weight if mod.rotated_weight is not None else mod.weight.data

        # For LSQS: smooth is applied to weight inside _forward_common. Since
        # rotated_weight is set only for Hadamard quantizers, fold smooth here
        # manually for LSQS variants.
        if hasattr(mod, 'x_Qparms') and 'smooth_log_scale' in mod.x_Qparms:
            smooth = torch.exp(mod.x_Qparms['smooth_log_scale'])
            folded = folded * smooth.view(1, -1)

        replacements[name] = _ExportQLinear(mod, folded)

    # Replace modules
    for name, new_mod in replacements.items():
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        leaf = parts[-1]
        if leaf.isdigit():
            parent[int(leaf)] = new_mod
        else:
            setattr(parent, leaf, new_mod)

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def export_vggt_to_onnx(
    checkpoint_path: str,
    output_path: str,
    args,
    layer_prefixes: list = None,
    num_frames: int = 4,
    image_size: int = 224,
    opset: int = 17,
    simplify: bool = True,
):
    """
    Loads a QAT VGGT checkpoint and exports it to ONNX with Q/DQ nodes.

    Args:
        checkpoint_path: Path to the QAT checkpoint (.pth).
        output_path:     Destination ONNX file path.
        args:            Namespace with quantizer config (num_bits, x_quantizer, etc.).
        layer_prefixes:  List of module name prefixes to quantize (default: QLAYERS).
        num_frames:      Fixed frame count for the dummy input.
        image_size:      Spatial resolution (H = W).
        opset:           ONNX opset version (17 recommended for Q/DQ support).
        simplify:        Run onnxsim.simplify after export.
    """
    if layer_prefixes is None:
        layer_prefixes = QLAYERS

    # 1. Build model + apply QAT structure
    with initialize(version_base=None, config_path='../../src/models/depth/vggt/training/config'):
        cfg = compose(config_name='default')
    trainer = Trainer(**cfg)
    model = trainer.model.module

    model = selective_quantize_layers(model, args, layer_prefixes)

    # 2. Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 3. Warm-up forward to populate Hadamard / smooth caches
    dummy = torch.randn(1, num_frames, 3, image_size, image_size, device=device)
    with torch.no_grad():
        model(images=dummy)

    # 4. Swap QLinear → ExportQLinear (extracts folded weights from cache)
    model = _replace_qlinear_with_export(model, layer_prefixes)
    model.eval()

    # 5. Export
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    dynamic_axes = {
        'images': {0: 'batch', 1: 'frames'},
        'output': {0: 'batch'},
    }
    torch.onnx.export(
        model,
        {'images': dummy},
        output_path,
        opset_version=opset,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )
    print(f'Exported ONNX model to {output_path}')

    # 6. Validate + optionally simplify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print('ONNX model validation passed.')

    if simplify:
        sim_model, ok = onnxsim.simplify(onnx_model)
        if ok:
            onnx.save(sim_model, output_path)
            print('ONNX model simplified.')
        else:
            print('onnxsim simplification failed — keeping original ONNX graph.')

    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   required=True)
    p.add_argument('--output',       default='deploy/engines/vggt_w8a8.onnx')
    p.add_argument('--num_bits',     type=int, default=8)
    p.add_argument('--x_quantizer', default='LSQ_quantizer')
    p.add_argument('--w_quantizer', default='LSQ_quantizer')
    p.add_argument('--initializer', default='NMSE_initializer')
    p.add_argument('--num_frames',  type=int, default=4)
    p.add_argument('--image_size',  type=int, default=224)
    p.add_argument('--no_simplify', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    export_vggt_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        args=args,
        num_frames=args.num_frames,
        image_size=args.image_size,
        simplify=not args.no_simplify,
    )
