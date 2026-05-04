"""
Latency and throughput benchmarks for quantized VGGT on Jetson.

Compares:
  - FP32 PyTorch eager
  - FP16 PyTorch eager
  - W8A8 PyTorch (_int_mm, from run_realquant.py)
  - TRT INT8 engine

Usage:
  python -m deploy.runtime.benchmark \
      --trt_engine   deploy/engines/vggt_w8a8.trt \
      --checkpoint   model_zoo/pytorchcv/vggt_w8a8_lsq.pth \
      --num_frames   4 \
      --warmup_runs  10 \
      --bench_runs   50
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import numpy as np
import torch

from deploy.runtime.trt_inference import TRTInferenceEngine


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ──────────────────────────────────────────────────────────────────────────────

def _warmup_and_time_pytorch(model, images, warmup=10, runs=50, label='PyTorch'):
    device = next(model.parameters()).device
    images = images.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(images=images)
        torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(images=images)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    _print_row(label, times)
    return times


def _warmup_and_time_trt(engine_path, images_np, warmup=10, runs=50, label='TRT'):
    with TRTInferenceEngine(engine_path) as eng:
        for _ in range(warmup):
            eng.infer({'images': images_np})

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            eng.infer({'images': images_np})
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    _print_row(label, times)
    return times


def _print_row(label, times_ms):
    median = np.median(times_ms)
    mean   = np.mean(times_ms)
    p95    = np.percentile(times_ms, 95)
    fps    = 1000 / median
    print(f'  {label:<35}  median={median:6.1f}ms  mean={mean:6.1f}ms  p95={p95:6.1f}ms  FPS={fps:.1f}')


def _get_model_size_mb(model):
    total = sum(p.numel() * p.element_size() for p in list(model.parameters()) + list(model.buffers()))
    return total / 1024 / 1024


# ──────────────────────────────────────────────────────────────────────────────
# Public benchmark entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    trt_engine_path: str = None,
    pytorch_fp32_model=None,
    pytorch_int8_model=None,
    num_frames: int = 4,
    image_size: int = 224,
    warmup_runs: int = 10,
    bench_runs: int = 50,
):
    """
    Runs the benchmark suite and prints a comparison table.

    Args:
        trt_engine_path:   Path to .trt engine (optional).
        pytorch_fp32_model: FP32 VGGT model (optional).
        pytorch_int8_model: W8A8 real-integer model from run_realquant (optional).
        num_frames:        Number of input frames.
    """
    dummy = torch.randn(1, num_frames, 3, image_size, image_size)
    dummy_np = dummy.numpy().astype(np.float32)

    print('\n' + '=' * 70)
    print('VGGT BENCHMARK — latency per inference call')
    print(f'  Input shape: {list(dummy.shape)}, warmup={warmup_runs}, runs={bench_runs}')
    print('=' * 70)

    results = {}

    if pytorch_fp32_model is not None:
        results['PyTorch FP32'] = _warmup_and_time_pytorch(
            pytorch_fp32_model, dummy, warmup_runs, bench_runs, 'PyTorch FP32')

    if pytorch_fp32_model is not None:
        fp16_model = pytorch_fp32_model.half().cuda()
        results['PyTorch FP16'] = _warmup_and_time_pytorch(
            fp16_model, dummy.half(), warmup_runs, bench_runs, 'PyTorch FP16')

    if pytorch_int8_model is not None:
        results['PyTorch W8A8 (_int_mm)'] = _warmup_and_time_pytorch(
            pytorch_int8_model, dummy, warmup_runs, bench_runs, 'PyTorch W8A8 (torch._int_mm)')

    if trt_engine_path and os.path.exists(trt_engine_path):
        results['TRT INT8'] = _warmup_and_time_trt(
            trt_engine_path, dummy_np, warmup_runs, bench_runs, 'TRT INT8')

    # Speedup summary
    if results:
        print('\nSpeedup summary (relative to fastest PyTorch baseline):')
        baselines = [k for k in ('PyTorch FP32', 'PyTorch FP16') if k in results]
        if baselines:
            base_median = np.median(results[baselines[0]])
            for label, times in results.items():
                speedup = base_median / np.median(times)
                print(f'  {label:<35}  {speedup:+.2f}×')

    if pytorch_fp32_model is not None:
        print(f'\n  FP32 model size:  {_get_model_size_mb(pytorch_fp32_model):.1f} MB')
    if pytorch_int8_model is not None:
        print(f'  W8A8 model size:  {_get_model_size_mb(pytorch_int8_model):.1f} MB')

    print('=' * 70 + '\n')
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--trt_engine',   default=None)
    p.add_argument('--checkpoint',   default=None, help='QAT checkpoint for W8A8 PyTorch baseline')
    p.add_argument('--num_bits',     type=int, default=8)
    p.add_argument('--x_quantizer', default='LSQ_quantizer')
    p.add_argument('--w_quantizer', default='LSQ_quantizer')
    p.add_argument('--num_frames',  type=int, default=4)
    p.add_argument('--image_size',  type=int, default=224)
    p.add_argument('--warmup_runs', type=int, default=10)
    p.add_argument('--bench_runs',  type=int, default=50)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    fp32_model = int8_model = None

    if args.checkpoint:
        from src.models.depth.vggt.training.trainer import Trainer
        from src.run_distill import selective_quantize_layers
        from src.run_realquant import QLAYERS, pack_model_to_real_int
        from hydra import initialize, compose

        with initialize(version_base=None, config_path='../../src/models/depth/vggt/training/config'):
            cfg = compose(config_name='default')
        trainer = Trainer(**cfg)
        fp32_model = trainer.model.module

        ckpt = torch.load(args.checkpoint, map_location='cpu')
        state = ckpt.get('model', ckpt)

        qat_model = selective_quantize_layers(fp32_model, args, QLAYERS)
        qat_model.load_state_dict(state, strict=False)
        int8_model = pack_model_to_real_int(qat_model, QLAYERS)
        int8_model = int8_model.cuda()

        fp32_model = trainer.model.module.cuda()

    run_benchmark(
        trt_engine_path=args.trt_engine,
        pytorch_fp32_model=fp32_model,
        pytorch_int8_model=int8_model,
        num_frames=args.num_frames,
        image_size=args.image_size,
        warmup_runs=args.warmup_runs,
        bench_runs=args.bench_runs,
    )
