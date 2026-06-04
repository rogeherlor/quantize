import os
import sys

os.environ.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)

# Both FP32 and FP16 use the single-pool strategy: pre-establish ONE large NvMap
# segment before any model allocation. All parameter + activation allocations draw
# from this segment — zero per-allocation NvMap calls, no handle-table exhaustion.
# expandable_segments (VMM not supported on Jetson NvMap) would create a separate
# NvMap handle per parameter; with 1797 params that exhausts the handle table before
# inference even starts, causing NVML_SUCCESS assertion failures in scratch_forward.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

# Drop kernel page cache before torch/CUDA initialise.
# CUDA runtime claims physical pages at init time; clearing the page cache first
# maximises contiguous free pages for the pool's chunked NvMap allocations.
try:
    with open("/proc/sys/vm/drop_caches", "w") as _dc:
        _dc.write("3\n")
except OSError:
    pass

# Compact physical memory before CUDA initialises.
# drop_caches frees pages but leaves them scattered; compact_memory coalesces them
# so the buddy allocator can serve the large contiguous requests NvMap needs.
try:
    with open("/proc/sys/vm/compact_memory", "w") as _cm:
        _cm.write("1\n")
    import time as _time_pre
    _time_pre.sleep(2)  # compaction is async; 2 s is sufficient for 7.6 GB
    del _time_pre
except OSError:
    pass

import argparse
import contextlib
import csv
import datetime
import gc
import glob
import json
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src/models/depth/vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ── Memory profiler ───────────────────────────────────────────────────────────

class MemoryProfiler:
    """
    Background-thread memory sampler for Jetson unified-RAM profiling.

    Polls /proc/meminfo and the CUDA allocator every poll_interval seconds.
    Supports named checkpoints (point-in-time snapshots) and writes two
    artifacts to profile_dir when save() is called:

      {dtype_tag}_memory_timeline.csv  — timestamped rows
      {dtype_tag}_summary.json         — peak values + per-checkpoint snapshots
                                         + swap_was_needed verdict
    """

    def __init__(self, profile_dir: str, dtype_tag: str, poll_interval: float = 0.5):
        self.profile_dir = profile_dir
        self.dtype_tag = dtype_tag
        self.poll_interval = poll_interval
        self._rows: list = []
        self._checkpoints: dict = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        os.makedirs(profile_dir, exist_ok=True)

    @staticmethod
    def _read_meminfo() -> dict:
        fields = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    fields[parts[0].rstrip(":")] = int(parts[1])
        return fields

    @staticmethod
    def _cuda_stats() -> dict:
        try:
            alloc   = torch.cuda.memory_allocated()   / 1024**2
            reserv  = torch.cuda.memory_reserved()    / 1024**2
            mx      = torch.cuda.max_memory_allocated()/ 1024**2
        except Exception:
            alloc = reserv = mx = 0.0
        return {
            "cuda_allocated_mb": round(alloc,  1),
            "cuda_reserved_mb":  round(reserv, 1),
            "cuda_max_alloc_mb": round(mx,     1),
        }

    def _sample(self) -> dict:
        mi = self._read_meminfo()
        cs = self._cuda_stats()
        return {
            "ts":               time.time(),
            "mem_free_mb":      round(mi.get("MemFree",      0) / 1024, 1),
            "mem_available_mb": round(mi.get("MemAvailable", 0) / 1024, 1),
            "swap_used_mb":     round((mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) / 1024, 1),
            "swap_total_mb":    round(mi.get("SwapTotal", 0) / 1024, 1),
            **cs,
        }

    def _poll_loop(self):
        while not self._stop.is_set():
            self._rows.append(self._sample())
            self._stop.wait(self.poll_interval)

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def checkpoint(self, name: str):
        snap = self._sample()
        snap["label"] = name
        self._checkpoints[name] = snap
        print(
            f"  [checkpoint:{name}] "
            f"CUDA={snap['cuda_allocated_mb']:.0f} MB alloc"
            f" | sys_free={snap['mem_free_mb']:.0f} MB"
            f" | swap_used={snap['swap_used_mb']:.0f} MB"
        )

    def save(self) -> dict:
        tag = self.dtype_tag
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = os.path.join(self.profile_dir, f"{tag}_memory_timeline.csv")
        if self._rows:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
                w.writeheader()
                w.writerows(self._rows)
            print(f"  Timeline CSV : {csv_path}  ({len(self._rows)} samples)")

        peak_cuda = max((r["cuda_allocated_mb"] for r in self._rows), default=0.0)
        peak_swap = max((r["swap_used_mb"]       for r in self._rows), default=0.0)
        min_free  = min((r["mem_free_mb"]         for r in self._rows), default=0.0)
        min_avail = min((r["mem_available_mb"]    for r in self._rows), default=0.0)
        total_ram = self._read_meminfo().get("MemTotal", 0) / 1024

        # Verdict: was NVMe swap actually consumed during inference?
        # 200 MB threshold accounts for background OS activity.
        swap_needed = peak_swap > 200
        dtype_label = self.dtype_tag.upper()
        rec = (
            f"{dtype_label} can run without NVMe swap — swap usage was minimal (<200 MB)."
            if not swap_needed else
            f"{dtype_label} requires NVMe swap. Peak swap={peak_swap:.0f} MB. "
            f"Min free RAM={min_free:.0f} MB. Keep swapfile active for production {dtype_label} runs."
        )

        summary = {
            "run_timestamp":        ts,
            "dtype":                tag,
            "total_ram_mb":         round(total_ram, 1),
            "peak_cuda_alloc_mb":   round(peak_cuda, 1),
            "peak_swap_used_mb":    round(peak_swap, 1),
            "min_sys_free_mb":      round(min_free,  1),
            "min_sys_available_mb": round(min_avail, 1),
            "swap_was_needed":      swap_needed,
            "recommendation":       rec,
            "checkpoints":          self._checkpoints,
        }

        json_path = os.path.join(self.profile_dir, f"{tag}_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Summary JSON : {json_path}")

        print("\n" + "=" * 60)
        print(f"  MEMORY PROFILE SUMMARY  ({tag.upper()})")
        print("=" * 60)
        print(f"  Total RAM              : {total_ram:.0f} MB")
        print(f"  Peak CUDA allocated    : {peak_cuda:.0f} MB")
        print(f"  Peak swap used         : {peak_swap:.0f} MB")
        print(f"  Min sys RAM free       : {min_free:.0f} MB")
        print(f"  Min sys RAM available  : {min_avail:.0f} MB")
        print(f"  Swap was needed        : {swap_needed}")
        print(f"  {rec}")
        print("=" * 60)

        return summary


# ── Image utilities ───────────────────────────────────────────────────────────

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB image → float32 CHW tensor in [0,1] without numpy."""
    buf = img.tobytes()
    t = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
    return t.reshape(img.height, img.width, 3).float().div(255.0).permute(2, 0, 1)


def _load_images_numpy_free(image_path_list, mode="crop"):
    """Replicates load_and_preprocess_images without any numpy dependency."""
    target_size = 518
    images = []
    shapes = set()

    for path in image_path_list:
        img = Image.open(path)
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        w, h = img.size

        if mode == "pad":
            if w >= h:
                nw = target_size
                nh = round(h * (nw / w) / 14) * 14
            else:
                nh = target_size
                nw = round(w * (nh / h) / 14) * 14
        else:
            nw = target_size
            nh = round(h * (nw / w) / 14) * 14

        img = img.resize((nw, nh), Image.Resampling.BICUBIC)
        t = _pil_to_tensor(img)

        if mode == "crop" and nh > target_size:
            start_y = (nh - target_size) // 2
            t = t[:, start_y: start_y + target_size, :]

        if mode == "pad":
            hp = target_size - t.shape[1]
            wp = target_size - t.shape[2]
            if hp > 0 or wp > 0:
                pt, pl = hp // 2, wp // 2
                t = F.pad(t, (pl, wp - pl, pt, hp - pt), value=1.0)

        shapes.add((t.shape[1], t.shape[2]))
        images.append(t)

    if len(shapes) > 1:
        mh = max(s[0] for s in shapes)
        mw = max(s[1] for s in shapes)
        padded = []
        for t in images:
            hp = mh - t.shape[1]
            wp = mw - t.shape[2]
            if hp > 0 or wp > 0:
                pt, pl = hp // 2, wp // 2
                t = F.pad(t, (pl, wp - pl, pt, hp - pt), value=1.0)
            padded.append(t)
        images = padded

    return torch.stack(images)


# ── Checkpoint loader ─────────────────────────────────────────────────────────

def _load_state_dict_streaming(model: nn.Module, path: str, dtype=torch.float16):
    """
    Load a (possibly FP32) checkpoint into the already-CUDA model without ever
    building a full CPU-side copy of the weights.

    mmap=True: the file is demand-paged — only pages for the tensor currently
    being processed enter physical RAM.  We immediately delete each entry from
    the dict after copying so the OS can evict the mmap page.
    Peak extra RAM: a few MB (one tensor at a time) instead of 2.4 GB.
    """
    print(f"  Loading checkpoint (streaming mmap, target dtype={dtype}) from {path} ...")
    state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=False)

    # Unwrap 'model' key if this is a full training checkpoint
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    param_map  = dict(model.named_parameters())
    buffer_map = dict(model.named_buffers())

    matched, unexpected = 0, []
    for key in list(state_dict.keys()):
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            del state_dict[key]
            continue
        if key in param_map:
            param_map[key].data.copy_(tensor)
            matched += 1
        elif key in buffer_map:
            buffer_map[key].data.copy_(tensor)
            matched += 1
        else:
            unexpected.append(key)
        del state_dict[key]   # release mmap page reference immediately

    del state_dict
    gc.collect()
    print(f"  Loaded {matched} tensors. "
          f"Unexpected: {len(unexpected)}{f' ({unexpected[:3]})' if unexpected else '.'}")
    print(f"  CUDA after checkpoint: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")


def _load_checkpoint_to_cuda(model: nn.Module, path: str, dtype: torch.dtype):
    """Stream checkpoint directly from NVMe mmap into the pre-established CUDA pool.

    The model must be built with torch.device('meta') so it has zero physical
    footprint. For each tensor in the checkpoint:
      1. An mmap page (a few MB max) is demand-paged from NVMe into CPU RAM.
      2. tensor.to('cuda', dtype) allocates from the caching pool — one
         contiguous sub-block carved from the large NvMap segment established
         at startup. No new NvMap call, no contiguity issue.
      3. The mmap reference is deleted; the OS immediately evicts the page.

    After all tensors are loaded, model.load_state_dict(assign=True) replaces
    the meta tensors with the real CUDA tensors.

    Non-persistent buffers (not saved in checkpoint) are fixed up afterwards:
    VGGT has exactly two — _resnet_mean and _resnet_std — in the Aggregator
    (aggregator.py line 139-140, shape [1,1,3,1,1]).
    """
    print(f"  Streaming checkpoint to CUDA pool ({dtype}) from {path} ...")
    state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    # Only load tensors that exist in this model (respects enable_point=False etc.).
    # Skipping disabled-head keys avoids ever allocating their weights to CUDA,
    # saving ~400-600 MB of pool space for inference activations.
    _model_keys = set(dict(model.named_parameters()).keys()) | \
                  set(dict(model.named_buffers()).keys())

    cuda_sd = {}
    n = 0
    skipped = 0
    for key in list(state_dict.keys()):
        tensor = state_dict[key]
        if isinstance(tensor, torch.Tensor):
            if key in _model_keys:
                cuda_sd[key] = tensor.to("cuda", dtype=dtype)
                n += 1
            else:
                skipped += 1
        del state_dict[key]   # release mmap page immediately
    del state_dict
    gc.collect()

    missing, unexpected = model.load_state_dict(cuda_sd, assign=True, strict=False)
    del cuda_sd
    gc.collect()

    print(f"  Loaded {n} tensors ({skipped} skipped — disabled heads). "
          f"Missing: {len(missing)}. Unexpected: {len(unexpected)}.")

    # Fix the two non-persistent buffers that aren't in the checkpoint.
    # Values are ImageNet normalization constants hardcoded in aggregator.py.
    for module in model.modules():
        for buf_name, val in (
            ("_resnet_mean", [0.485, 0.456, 0.406]),
            ("_resnet_std",  [0.229, 0.224, 0.225]),
        ):
            buf = getattr(module, buf_name, None)
            if buf is not None and (buf.is_meta or buf.device.type != "cuda"):
                module.register_buffer(
                    buf_name,
                    torch.tensor(val, dtype=dtype, device="cuda").view(1, 1, 3, 1, 1),
                    persistent=False,
                )

    print(f"  CUDA after checkpoint: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")


def _to_cuda_incremental(model: nn.Module, device: str, dtype: torch.dtype):
    """
    Move model to CUDA one parameter at a time, freeing each CPU tensor
    before allocating the next CUDA one.

    model.to('cuda') holds all CPU parameter references alive through its
    recursive _apply() traversal. On Jetson with ~100 MB sys_free, that means
    NvMap can't reclaim the freed CPU pages fast enough and fails with ENOMEM.

    Here we iterate by name, replace in the parent module immediately, then
    drop the local reference and call gc.collect(). Large tensors (>128 KB)
    are allocated via mmap, so del → munmap → pages returned to OS instantly,
    making them available for the very next NvMap allocation.
    """
    # Collect names only (no live tensor references in the list).
    param_names  = [n for n, _ in model.named_parameters()]
    buffer_names = [n for n, _ in model.named_buffers()]

    total = len(param_names) + len(buffer_names)
    done  = 0

    for name in param_names:
        parent, leaf = (model.get_submodule(name.rsplit('.', 1)[0]), name.rsplit('.', 1)[1]) \
                       if '.' in name else (model, name)
        param = parent._parameters[leaf]
        if param is None:
            continue
        cuda_data = param.data.to(device, dtype=dtype)
        parent._parameters[leaf] = nn.Parameter(cuda_data, requires_grad=param.requires_grad)
        del param       # drop last CPU reference → munmap → pages back to OS
        gc.collect()
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{total} tensors → CUDA  "
                  f"({torch.cuda.memory_allocated() / 1024**2:.0f} MB allocated)")

    for name in buffer_names:
        parent, leaf = (model.get_submodule(name.rsplit('.', 1)[0]), name.rsplit('.', 1)[1]) \
                       if '.' in name else (model, name)
        buf = parent._buffers[leaf]
        if buf is None:
            continue
        parent._buffers[leaf] = buf.to(device, dtype=dtype)
        del buf
        gc.collect()
        done += 1

    print(f"  {done}/{total} tensors → CUDA  "
          f"({torch.cuda.memory_allocated() / 1024**2:.0f} MB allocated)")


# ── Argument parser ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="VGGT standalone inference with memory profiling")
    prec = p.add_mutually_exclusive_group()
    prec.add_argument(
        "--fp32", action="store_true",
        help="Run in the checkpoint's native precision (no dtype conversion). "
             "The vggt_1B_commercial.pt checkpoint is FP32, so this needs NVMe swap.",
    )
    prec.add_argument(
        "--fp16", action="store_true",
        help="Run in half precision (default — safe for Jetson 7.8 GB unified RAM).",
    )
    p.add_argument(
        "--profile", action="store_true",
        help="Enable comprehensive memory profiling (MemoryProfiler + torch.profiler).",
    )
    p.add_argument(
        "--profile_dir", default="data3/rogelio/profiles",
        help="Output directory for profile artifacts.",
    )
    p.add_argument(
        "--model_path", default="data3/rogelio/model_zoo/vggt/vggt_1B_commercial.pt",
    )
    p.add_argument(
        "--image_dir", default="src/models/depth/vggt/examples/kitchen/images",
    )
    p.add_argument(
        "--max_frames", type=int, default=None,
        help="Max input frames. Defaults to 2 for --fp32 (FP32 activation budget), 10 for FP16.",
    )
    p.add_argument(
        "--dpt_chunk", type=int, default=None,
        help="Frames per DPT decoder chunk. Smaller = less activation memory, more compute passes. "
             "Defaults to 2 for --fp32, 4 for FP16.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

args = _parse_args()
if args.max_frames is None:
    args.max_frames = 2 if args.fp32 else 10
if args.dpt_chunk is None:
    args.dpt_chunk = 2 if args.fp32 else 4

MODEL_PATH = args.model_path
IMAGE_DIR  = args.image_dir
MAX_FRAMES = args.max_frames
DPT_CHUNK  = args.dpt_chunk

device = "cuda" if torch.cuda.is_available() else "cpu"
# --fp32: use the checkpoint's original precision (FP32 for vggt_1B_commercial.pt).
# Default: FP16 — Jetson 7.8 GB unified RAM, FP32 weights (4.8 GB) need NVMe swap.
dtype     = torch.float32 if args.fp32 else torch.float16
dtype_tag = "fp32"         if args.fp32 else "fp16"

print(f"Device: {device}, dtype: {dtype}  ({'original checkpoint precision' if args.fp32 else 'half precision'})")
print(f"PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")

if args.fp32:
    _mi = {l.split()[0].rstrip(":"): int(l.split()[1])
           for l in open("/proc/meminfo") if len(l.split()) >= 2}
    _swap_used_mb  = (_mi["SwapTotal"] - _mi["SwapFree"]) // 1024
    _swap_total_mb = _mi["SwapTotal"] // 1024
    print(f"  Full-precision mode: swap {_swap_used_mb} MB used / {_swap_total_mb} MB total")
    # FP32 needs a one-time NvMap pool of ~5 GB established early (while sys_free
    # is high). This requires the kernel to swap out CPU pages to provide NvMap
    # with physical RAM. Zram (RAM-backed, ~3.8 GB) cannot extend physical RAM —
    # only NVMe-backed swap can. Without it, the pool fails, the NvMap handle table
    # fills during the incremental param transfer, and inference OOMs.
    if _mi["SwapTotal"] < 6 * 1024 * 1024:   # less than 6 GB → no NVMe swap active
        print("""
  ERROR: Full-precision inference requires NVMe-backed swap (≥6 GB).
  Only {_swap_total_mb} MB swap detected — this is zram (RAM-backed, cannot extend physical RAM).

  On the Jetson HOST (or inside the privileged container), run:
      sudo bash scripts/setup_fp32_swap.sh

  Then re-run this script.
""".format(_swap_total_mb=_swap_total_mb))
        sys.exit(1)

print(f"Free before model: {torch.cuda.mem_get_info()[0] / 1024**2:.0f} MB")

# ── Start profiler ────────────────────────────────────────────────────────────
mem_profiler = None
if args.profile:
    mem_profiler = MemoryProfiler(args.profile_dir, dtype_tag=dtype_tag)
    mem_profiler.__enter__()
    mem_profiler.checkpoint("script_start")

# ── Pre-establish CUDA caching pool ──────────────────────────────────────────
# Second compact_memory pass: PyTorch's CUDA context init (above) allocates
# several hundred MB and re-fragments physical pages. Compacting here, after
# CUDA is fully initialised and idle, gives NvMap the largest possible
# contiguous regions for pool establishment.
try:
    with open("/proc/sys/vm/compact_memory", "w") as _cm2:
        _cm2.write("1\n")
    time.sleep(3)
except OSError:
    pass

_model_needs_mb = 4200 if args.fp32 else 2400  # FP32: ~4793 full - ~600 disabled heads
_cuda_free, _ = torch.cuda.mem_get_info()
_cuda_free_mb = _cuda_free // (1024**2)

def _alloc_pool(target_mb, device, min_chunk_mb=32):
    chunks, pooled = [], 0
    chunk = 512
    while pooled < target_mb and chunk >= min_chunk_mb:
        want = min(chunk, target_mb - pooled)
        try:
            chunks.append(torch.empty(want * 1024 * 1024 // 4, dtype=torch.float32, device=device))
            pooled += want
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            chunk //= 2
    return chunks, pooled

if args.fp32:
    # FP32: single pool covering full model. All available CUDA memory minus a small
    # runtime reserve. Without this, checkpoint streaming overflows the pool and
    # PyTorch tries NvMap expansion with <100 MB physical free → assert.
    _target_mb = max(0, _cuda_free_mb - 50)
    print(f"  Pre-allocating CUDA pool: target {_target_mb} MB ...")
    _chunks, _pooled_mb = _alloc_pool(_target_mb, device)
    del _chunks
    gc.collect()
    _act_chunks = None
else:
    # FP16 two-phase allocation — solves the fragmentation problem:
    #
    # A single pool (model + activations) gets fragmented by the 1797 model param
    # tensors scattered through it. empty_cache() + warmup can only recover ~574 MB
    # of contiguous free space, far less than the 1200 MB needed for 10-frame inference.
    #
    # Two-phase fix:
    #   Phase 1 — model pool kept ALIVE while phase 2 runs. The caching allocator
    #   free list is empty, so phase 2 is forced to call NvMap for fresh segments.
    #   Phase 2 — activation pool from those fresh NvMap segments (never fragmented
    #   by model params because they aren't allocated yet).
    #   Release model pool → model loads from those segments (fragments them).
    #   Release activation pool AFTER model load → contiguous free blocks for inference.
    #
    # Empirical activation scaling (Flash Attention → linear with S):
    #   6 frames → 654 MB, 8 frames → 937 MB → ~130 MB/frame.
    _model_pool_target = _model_needs_mb + 300          # 2700 MB (300 MB buffer)
    _act_pool_target   = max(512, MAX_FRAMES * 130 + 300)  # e.g. 1600 MB for 10 frames

    print(f"  Pre-allocating model pool:      target {_model_pool_target} MB ...")
    _model_chunks, _model_pooled_mb = _alloc_pool(_model_pool_target, device)

    print(f"  Pre-allocating activation pool: target {_act_pool_target} MB "
          f"(separate NvMap segments — model pool still alive) ...")
    _act_chunks, _act_pooled_mb = _alloc_pool(_act_pool_target, device)

    # Release model pool to free list → model will allocate from these segments.
    del _model_chunks
    gc.collect()
    _pooled_mb = _model_pooled_mb + _act_pooled_mb

_reserved_mb = torch.cuda.memory_reserved() // (1024**2)
if args.fp32:
    print(f"  Pool ready: {_reserved_mb} MB reserved  ({_pooled_mb} MB established, model needs ~{_model_needs_mb} MB)")
else:
    print(f"  Pool ready: {_reserved_mb} MB reserved  "
          f"(model {_model_pooled_mb} MB + activation {_act_pooled_mb} MB, model needs ~{_model_needs_mb} MB)")
if mem_profiler:
    mem_profiler.checkpoint("after_pool_establish")

# ── Build model ───────────────────────────────────────────────────────────────
# Both FP32 and FP16 use meta device — zero physical footprint during construction.
# The pool above is already in the caching allocator's free list as one
# physically-contiguous NvMap segment. _load_checkpoint_to_cuda streams each
# tensor from the NVMe mmap directly into that segment — no CPU staging needed.
# VGGT's constructor calls .item() on a meta linspace tensor
# (vision_transformer.py:119) to compute stochastic depth drop-path rates.
# Patch .item() to return 0 for meta tensors: the values are irrelevant because
# model.eval() disables StochasticDepth in the forward pass.
print("Building model structure (meta device, zero physical RAM) ...")
_orig_item = torch.Tensor.item
torch.Tensor.item = lambda self: 0 if self.is_meta else _orig_item(self)
try:
    with torch.device("meta"):
        if args.fp32:
            # FP32: pool free after model load = ~960 MB. Disable point/track heads
            # (~400-600 MB) to leave enough room for DPT inference activations.
            model = VGGT(enable_point=False, enable_track=False).eval()
        else:
            # FP16: pool free after model load = ~3000+ MB. All heads fit.
            model = VGGT().eval()
finally:
    torch.Tensor.item = _orig_item
print("  Meta model built.")
if mem_profiler:
    mem_profiler.checkpoint("after_model_init")

_load_checkpoint_to_cuda(model, MODEL_PATH, dtype)
if mem_profiler:
    mem_profiler.checkpoint("after_model_load")

print(f"  CUDA allocated: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

if not args.fp32:
    # FP16: release the activation pool now that the model is loaded.
    # These blocks came from fresh NvMap segments allocated before model load,
    # so they are fully contiguous and unfragmented by model parameter allocations.
    del _act_chunks
    gc.collect()
    print(f"  Activation pool released: {_act_pooled_mb} MB contiguous free for inference")
else:
    # FP32: warmup defragments the single pool's free list after model load.
    # The 1341 model param tensors leave small gaps; empty_cache() merges them
    # back into NvMap so the warmup can re-acquire them as larger contiguous blocks.
    torch.cuda.empty_cache()
    gc.collect()
    free_before_warmup, total = torch.cuda.mem_get_info()
    print(f"  CUDA free before warmup: {free_before_warmup / 1024**2:.0f} MB / {total / 1024**2:.0f} MB total")
    _target_warmup_mb = max(0, free_before_warmup // (1024**2) - 50)
    _warmup_chunks = []
    _warmed_mb = 0
    _chunk_mb = _target_warmup_mb
    while _warmed_mb < _target_warmup_mb and _chunk_mb >= 64:
        _want = min(_chunk_mb, _target_warmup_mb - _warmed_mb)
        try:
            _warmup_chunks.append(
                torch.empty(_want * 1024 * 1024 // 4, dtype=torch.float32, device=device)
            )
            _warmed_mb += _want
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            _chunk_mb //= 2
    del _warmup_chunks
    gc.collect()
    print(f"  Allocator free list pre-warmed with {_warmed_mb} MB")

# ── Load images ───────────────────────────────────────────────────────────────
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)
image_paths = image_paths[:MAX_FRAMES]
print(f"Found {len(image_paths)} images (capped at {MAX_FRAMES}): {[os.path.basename(p) for p in image_paths]}")
images = _load_images_numpy_free(image_paths).to(device, dtype=dtype)
print(f"Input shape: {images.shape}, dtype={images.dtype}")

if mem_profiler:
    mem_profiler.checkpoint("after_preprocessing")

# ── Inference ─────────────────────────────────────────────────────────────────
print("Running inference ...")

if mem_profiler:
    mem_profiler.checkpoint("inference_start")

_prof_ctx = (
    torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
        with_stack=False,   # stack traces inflate trace size significantly on a 1.2B model
    )
    if args.profile
    else contextlib.nullcontext()
)

with torch.inference_mode(), _prof_ctx as _prof:
    predictions = model(images.unsqueeze(0), frames_chunk_size=DPT_CHUNK)   # B=1

if mem_profiler:
    mem_profiler.checkpoint("inference_end")

if args.profile and _prof is not None:
    _trace_path = os.path.join(args.profile_dir, f"{dtype_tag}_torch_profile.json")
    _prof.export_chrome_trace(_trace_path)
    print(f"  Torch profiler trace : {_trace_path}")
    print(_prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))

# ── Decode poses ───────────────────────────────────────────────────────────────
extrinsic, intrinsic = pose_encoding_to_extri_intri(
    predictions["pose_enc"], images.shape[-2:]
)
predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic

print("\n--- Outputs ---")
for k, v in predictions.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {tuple(v.shape)}, dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v)}")

peak_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"\nPeak CUDA memory: {peak_mb:.0f} MB")

# ── Finalise profiler ─────────────────────────────────────────────────────────
if mem_profiler:
    mem_profiler.__exit__(None, None, None)
    mem_profiler.save()

print("Done.")
