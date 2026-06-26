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
import json
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _device_cc():
    """Compute capability of the build GPU as 'sm_87', or None if undetectable.

    Stamped into the engine sidecar so the runtime can refuse a foreign engine
    (TRT engines are not portable across GPU architectures)."""
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f'sm_{major}{minor}'
    except Exception:
        pass
    try:
        import pycuda.driver as cuda
        cuda.init()
        major, minor = cuda.Device(0).compute_capability()
        return f'sm_{major}{minor}'
    except Exception:
        return None


def _write_sidecar(output_path, trt, precision, input_shape):
    """Write <engine>.json describing the build (arch / TRT version / precision /
    shape) so trt_inference.py can validate before deserializing."""
    info = {
        'compute_capability': _device_cc(),
        'trt_version': trt.__version__,
        'precision': precision,
        'input_shape': [int(d) for d in input_shape],
        'built_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(output_path + '.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f'Wrote engine sidecar {output_path}.json ({info["compute_capability"]}, '
          f'TRT {info["trt_version"]})')


def _restrict_int8_layers(trt, network, config, allow_substrings):
    """Pin every MatMul/Conv layer NOT matching `allow_substrings` to FP16, and
    force the matching ones to INT8, so automatic PTQ quantizes exactly the same
    layer set as the LSQ/QDQ path. Requires OBEY_PRECISION_CONSTRAINTS.

    TRT layer names are derived from the ONNX node names, which carry the original
    module path (e.g. '...frame_blocks.6.attn.qkv/MatMul'). If fusion renames a
    layer, it falls through to FP16 — verification step 7 dumps the final set so
    the substrings can be adjusted.
    """
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    compute_types = {trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.CONVOLUTION}
    n_int8 = n_fp16 = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type not in compute_types:
            continue
        matched = any(s in layer.name for s in allow_substrings)
        try:
            if matched:
                layer.precision = trt.int8
                n_int8 += 1
            else:
                layer.precision = trt.float16
                n_fp16 += 1
        except Exception as e:  # some fused layers reject explicit precision
            print(f'  precision pin skipped for {layer.name}: {e}')
    print(f'Restricted INT8 to {n_int8} matched compute layers; pinned {n_fp16} to FP16.')


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
    calibrator=None,
    restrict_int8_to=None,
    timing_cache: str = None,
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

    # parse_from_file (not parse(f.read())) so TRT resolves >2 GB external-data
    # weight files relative to the ONNX's own directory; parsing from an in-memory
    # buffer makes TRT look in the CWD and fail with "Failed to open file: <weight>".
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(f'ONNX parse error [{i}]: {parser.get_error(i)}')
        raise RuntimeError('Failed to parse ONNX model.')
    print(f'Parsed ONNX model: {onnx_path}')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gb * (1 << 30)),
    )

    # Persistent timing cache: kernel-tactic timings are GPU-specific, so this is
    # tied to the Orin and dramatically speeds up rebuilds of the same/similar graph
    # (e.g. the FP16 → INT8 variants share most layers).
    cache_obj = None
    if timing_cache:
        buf = b''
        if os.path.exists(timing_cache):
            with open(timing_cache, 'rb') as f:
                buf = f.read()
            print(f'Loaded timing cache {timing_cache} ({len(buf)} bytes)')
        cache_obj = config.create_timing_cache(buf)
        config.set_timing_cache(cache_obj, ignore_mismatch=True)

    # Precision flags
    if precision == 'fp32':
        pass  # default; no flag
    elif precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)   # fallback for unsupported INT8 ops
        if calibrator is not None:
            # Automatic PTQ: TRT estimates scales from calibration data.
            config.int8_calibrator = calibrator
        # else: Q/DQ nodes in the ONNX carry the (LSQ) scales — no calibrator needed.
    elif precision == 'int4':
        # Experimental: TRT 10.x weight-only INT4 (W4A16)
        config.set_flag(trt.BuilderFlag.INT4)
        config.set_flag(trt.BuilderFlag.FP16)

    # Restrict INT8 to a specific set of layers (the QLAYERS Linears) so the
    # automatic-PTQ engine quantizes the SAME layers as the LSQ/QDQ engine.
    # Every MatMul/Conv whose name does not match is pinned to FP16.
    if precision == 'int8' and restrict_int8_to:
        _restrict_int8_layers(trt, network, config, restrict_int8_to)

    # Optimization profile: only needed when the input has a dynamic dim.
    # Our pose ONNX is fixed-shape [1,10,3,350,518] → skip the profile.
    inp = network.get_input(0)
    if any(d < 0 for d in inp.shape):
        profile = builder.create_optimization_profile()
        profile.set_shape(
            inp.name,
            min=(1, min_frames, 3, image_size, image_size),
            opt=(1, opt_frames, 3, image_size, image_size),
            max=(1, max_frames, 3, image_size, image_size),
        )
        config.add_optimization_profile(profile)

    # Build and serialize
    print(f'Building TRT engine ({precision.upper()}) — this may take several minutes...')
    input_shape = tuple(inp.shape)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError('TRT engine build failed.')

    # Persist the (now-populated) timing cache for faster subsequent builds.
    if timing_cache and cache_obj is not None:
        with open(timing_cache, 'wb') as f:
            f.write(cache_obj.serialize())
        print(f'Saved timing cache → {timing_cache}')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(engine_bytes)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'Engine saved to {output_path} ({size_mb:.1f} MB)')
    _write_sidecar(output_path, trt, precision, input_shape)
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
    p.add_argument('--timing_cache', default='deploy/artifacts/timing_cache/orin_sm87.cache',
                   help='Persistent TRT timing cache (Orin-specific) for fast rebuilds')
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
            timing_cache=args.timing_cache,
        )
