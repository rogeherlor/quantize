"""
TensorRT engine builder for quantized VGGT ONNX models.

Converts an ONNX model (with Q/DQ nodes) to a serialized TensorRT engine
(.trt file) with INT8 precision and Jetson-appropriate optimization profiles.

Usage:
  python -m deploy.export.trt_builder \
      --onnx  deploy/engines/vggt_w8a8.onnx \
      --output deploy/engines/vggt_w8a8.trt \
      --precision int8 \
      --min_frames 1 --opt_frames 4 --max_frames 8
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def build_engine(
    onnx_path: str,
    output_path: str,
    precision: str = 'int8',
    min_frames: int = 1,
    opt_frames: int = 4,
    max_frames: int = 8,
    image_size: int = 224,
    workspace_gb: float = 4.0,
    verbose: bool = False,
):
    """
    Builds a TensorRT engine from an ONNX model.

    Optimization profiles cover the dynamic 'frames' dimension of VGGT:
        min shape: (1, min_frames, 3, image_size, image_size)
        opt shape: (1, opt_frames, 3, image_size, image_size)
        max shape: (1, max_frames, 3, image_size, image_size)

    The batch dimension is fixed to 1 (Jetson Orin Nano, demo use case).

    Args:
        onnx_path:   Path to ONNX model (with Q/DQ nodes for INT8).
        output_path: Destination .trt engine file.
        precision:   'fp32', 'fp16', or 'int8'.
        min/opt/max_frames: Optimization profile bounds for dynamic frame dim.
        workspace_gb: GPU memory budget for TRT optimizer (GB).
        verbose:     Enable TRT verbose logging.
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            'TensorRT not found. Install via JetPack or:\n'
            '  sudo apt-get install python3-libnvinfer-dev'
        )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f'ONNX parse error [{i}]: {parser.get_error(i)}')
            raise RuntimeError('Failed to parse ONNX model.')
    print(f'Parsed ONNX model: {onnx_path}')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gb * (1 << 30)),
    )

    # Precision flags
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        # Q/DQ nodes in the ONNX graph carry scale information — TRT uses them
        # directly for INT8 calibration when the model was exported from a QAT
        # checkpoint. No external calibrator is needed.
        config.set_flag(trt.BuilderFlag.FP16)   # fallback for unsupported INT8 ops
    elif precision == 'int4':
        # Experimental: TRT 10.x weight-only INT4 (W4A16)
        config.set_flag(trt.BuilderFlag.INT4)
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimization profile: dynamic frames dimension (axis 1 of input)
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,
        min=(1, min_frames, 3, image_size, image_size),
        opt=(1, opt_frames, 3, image_size, image_size),
        max=(1, max_frames, 3, image_size, image_size),
    )
    config.add_optimization_profile(profile)

    # Build and serialize
    print(f'Building TRT engine ({precision.upper()}) — this may take several minutes...')
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError('TRT engine build failed.')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(engine_bytes)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'Engine saved to {output_path} ({size_mb:.1f} MB)')
    return output_path


def build_via_trtexec(
    onnx_path: str,
    output_path: str,
    precision: str = 'int8',
    min_frames: int = 1,
    opt_frames: int = 4,
    max_frames: int = 8,
    image_size: int = 224,
):
    """
    Alternative: shell out to trtexec for engine building.

    trtexec is simpler and gives useful timing output, but offers less
    programmatic control than the Python TRT API above.
    """
    import subprocess
    shape = f'1x{{f}}x3x{image_size}x{image_size}'
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={output_path}',
        f'--minShapes=images:{shape.format(f=min_frames)}',
        f'--optShapes=images:{shape.format(f=opt_frames)}',
        f'--maxShapes=images:{shape.format(f=max_frames)}',
        '--fp16',
    ]
    if precision == 'int8':
        cmd.append('--int8')
    elif precision == 'int4':
        cmd.append('--int4')
    cmd.append('--verbose')
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return output_path


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx',        required=True)
    p.add_argument('--output',      default='deploy/engines/vggt_w8a8.trt')
    p.add_argument('--precision',   choices=['fp32', 'fp16', 'int8', 'int4'], default='int8')
    p.add_argument('--min_frames',  type=int, default=1)
    p.add_argument('--opt_frames',  type=int, default=4)
    p.add_argument('--max_frames',  type=int, default=8)
    p.add_argument('--image_size',  type=int, default=224)
    p.add_argument('--workspace_gb', type=float, default=4.0)
    p.add_argument('--trtexec',     action='store_true', help='Use trtexec shell command instead of Python API')
    p.add_argument('--verbose',     action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    if args.trtexec:
        build_via_trtexec(
            onnx_path=args.onnx,
            output_path=args.output,
            precision=args.precision,
            min_frames=args.min_frames,
            opt_frames=args.opt_frames,
            max_frames=args.max_frames,
            image_size=args.image_size,
        )
    else:
        build_engine(
            onnx_path=args.onnx,
            output_path=args.output,
            precision=args.precision,
            min_frames=args.min_frames,
            opt_frames=args.opt_frames,
            max_frames=args.max_frames,
            image_size=args.image_size,
            workspace_gb=args.workspace_gb,
            verbose=args.verbose,
        )
