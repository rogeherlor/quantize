import os
import json
import datetime
import argparse
from src.logger import logger

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from functools import partial

from hydra import initialize, compose
import src.models.depth.vggt.vggt
from src.models.depth.vggt.training.trainer import Trainer

from src.utils import *
from src.scheduler_optimizer_class import scheduler_optimizer_class
import src.module_quantization as Q
from src.make_ex_name import *

from src.models.depth.vggt.evaluation.test_co3d import *

def _default_dpt_chunk():
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb < 10:
            logger.info(f"[Jetson] Detected {vram_gb:.1f} GB VRAM — using DPT frames_chunk_size=2 "
                        f"(vs default 8) to fit 10-frame activations in unified memory.")
            return 2
    return 8

def run_evaluation_vggt(model, model_path=None, frames_chunk_size=None, results_path=None,
                        categories=None, max_seqs=None, num_frames=10):
    """Main function to evaluate VGGT on CO3D dataset. Original vggt/evaluation/test_co3d.py

    categories: optional list of category names to restrict evaluation to (e.g.
        ["apple"]) for a fast memory/latency estimate. None → all 41 categories.
    max_seqs:   optional cap on sequences evaluated per category (None → all).
    num_frames: frames sampled per sequence. MUST match the frame count a fixed-shape
        TRT engine was built for (e.g. 6 for a 6-frame engine), else the token shape
        won't match the engine's `tokens` binding.
    """
    if frames_chunk_size is None:
        frames_chunk_size = _default_dpt_chunk()
    else:
        logger.info(f"DPT frames_chunk_size={frames_chunk_size} (explicit)")
    args = argparse.Namespace()
    args.debug = False
    args.use_ba = False
    args.fast_eval = False
    args.min_num_images = 50
    args.num_frames = num_frames
    args.co3d_dir = "data3/rogelio/co3d/dataset/"
    args.co3d_anno_dir = "data3/rogelio/co3d/preprocessed_dataset"
    args.seed = 0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sm_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 99
    if vram_gb < 10:
        # Jetson: model is loaded as FP16. Using BF16 autocast with FP16 weights
        # makes PyTorch cast each weight tensor to BF16 via cuDNN workspace
        # allocations that bypass the caching allocator → NvMap ENOMEM assert.
        # FP16 autocast on a FP16 model is a no-op: no casting, no extra allocs.
        dtype = torch.float16
        logger.info(f"Evaluation dtype: torch.float16 (Jetson {vram_gb:.1f} GB — matching model dtype, avoids BF16 cast overhead)")
    else:
        dtype = torch.bfloat16 if sm_major >= 8 else torch.float16
        logger.info(f"Evaluation dtype: {dtype} (SM{sm_major}.x — {'bfloat16 supported' if sm_major >= 8 else 'fallback to float16'})")

    # Set random seeds
    set_random_seeds(args.seed)

    # Record model weight memory before any forward pass.
    weight_mem_mb = torch.cuda.memory_allocated() // (1024 ** 2)
    logger.info(f"Model weight memory: {weight_mem_mb} MB")

    # Warmup: one synthetic forward pass using real evaluation dimensions
    # (350×518, patch_size=14 → 25×37 patches, 10 frames).
    # Triggers cuDNN algorithm selection and CUDA kernel JIT compilation so
    # the first timed sequence is not inflated by one-time setup costs.
    logger.info("Warming up CUDA kernels (1 synthetic forward pass) ...")
    _dummy = torch.zeros(1, args.num_frames, 3, 350, 518, dtype=dtype, device=device)
    with torch.inference_mode():
        model(_dummy, frames_chunk_size=frames_chunk_size)
    del _dummy
    torch.cuda.reset_peak_memory_stats()
    logger.info("Warmup done. Resetting peak memory stats — timing starts now.")

    # Categories to evaluate
    SEEN_CATEGORIES = [
        "apple", "backpack", "banana", "baseballbat", "baseballglove",
        "bench", "bicycle", "bottle", "bowl", "broccoli",
        "cake", "car", "carrot", "cellphone", "chair",
        "cup", "donut", "hairdryer", "handbag", "hydrant",
        "keyboard", "laptop", "microwave", "motorcycle", "mouse",
        "orange", "parkingmeter", "pizza", "plant", "stopsign",
        "teddybear", "toaster", "toilet", "toybus", "toyplane",
        "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
    ]

    if args.debug:
        SEEN_CATEGORIES = ["parkingmeter"]

    if categories:
        requested = [c for c in categories if c in SEEN_CATEGORIES]
        missing = [c for c in categories if c not in SEEN_CATEGORIES]
        if missing:
            logger.info(f"Ignoring unknown categories: {missing}")
        SEEN_CATEGORIES = requested or SEEN_CATEGORIES
        logger.info(f"Restricting evaluation to categories: {SEEN_CATEGORIES}")
    if max_seqs:
        logger.info(f"Capping evaluation to {max_seqs} sequence(s) per category")

    per_category_results = {}
    _seq_times = []
    _eval_start = __import__("time").perf_counter()

    for category in SEEN_CATEGORIES:
        logger.info(f"Loading annotation for {category} test set")
        annotation_file = os.path.join(args.co3d_anno_dir, f"{category}_test.jgz")

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            logger.info(f"Annotation file not found: {os.path.abspath(annotation_file)}")
            continue

        rError = []
        tError = []

        seq_names = sorted(list(annotation.keys()))
        if args.fast_eval and len(seq_names)>=10:
            seq_names = random.sample(seq_names, 10)
        seq_names = sorted(seq_names)
        if max_seqs:
            seq_names = seq_names[:max_seqs]


        logger.info("Testing Sequences: ")
        logger.info(seq_names)

        for seq_name in seq_names:
            seq_data = annotation[seq_name]
            logger.info("-" * 50)
            logger.info(f"Processing {seq_name} for {category} test set")
            if args.debug and not os.path.exists(os.path.join(args.co3d_dir, category, seq_name)):
                logger.info(f"Skipping {seq_name} (not found)")
                continue

            import time as _time
            torch.cuda.reset_peak_memory_stats()
            # Bracket the timer with cuda.synchronize() so perf_counter captures
            # COMPLETED GPU work, not just kernel-launch time. (CUDA events aren't
            # used here because process_sequence also runs CPU-side bundle
            # adjustment — events would time only the GPU stream. Pure-GPU compute
            # latency is reported separately via trtexec --noDataTransfers.)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = _time.perf_counter()
            seq_rError, seq_tError = process_sequence(
                model, seq_name, seq_data, category, args.co3d_dir,
                args.min_num_images, args.num_frames, args.use_ba, device, dtype,
                frames_chunk_size=frames_chunk_size,
                clear_cache=(frames_chunk_size >= 8),  # False on Jetson: empty_cache destroys activation pool
                use_autocast=(frames_chunk_size >= 8),  # False on Jetson: autocast queries NVML → assert
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _seq_s = _time.perf_counter() - _t0
            _peak_alloc_mb = torch.cuda.max_memory_allocated() // (1024 ** 2)
            _act_peak_mb = _peak_alloc_mb - weight_mem_mb
            if seq_rError is not None:
                logger.info(f"  time={_seq_s:.1f}s  peak_cuda={_peak_alloc_mb}MB  activations={_act_peak_mb}MB")
                _seq_times.append(_seq_s)

            logger.info("-" * 50)

            if seq_rError is not None and seq_tError is not None:
                rError.extend(seq_rError)
                tError.extend(seq_tError)

        if not rError:
            logger.info(f"No valid sequences found for {category}, skipping")
            continue

        rError = np.array(rError)
        tError = np.array(tError)

        Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
        Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
        Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)

        per_category_results[category] = {
            "rError": rError,
            "tError": tError,
            "Auc_30": Auc_30,
            "Auc_15": Auc_15,
            "Auc_5": Auc_5,
            "Auc_3": Auc_3
        }

        logger.info("="*80)
        # Print results with colors
        GREEN = "\033[92m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        logger.info(f"{BOLD}{BLUE}AUC of {category} test set:{RESET} {GREEN}{Auc_30:.4f} (AUC@30), {Auc_15:.4f} (AUC@15), {Auc_5:.4f} (AUC@5), {Auc_3:.4f} (AUC@3){RESET}")
        mean_AUC_30_by_now = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
        mean_AUC_15_by_now = np.mean([per_category_results[category]["Auc_15"] for category in per_category_results])
        mean_AUC_5_by_now = np.mean([per_category_results[category]["Auc_5"] for category in per_category_results])
        mean_AUC_3_by_now = np.mean([per_category_results[category]["Auc_3"] for category in per_category_results])
        logger.info(f"{BOLD}{BLUE}Mean AUC of categories by now:{RESET} {RED}{mean_AUC_30_by_now:.4f} (AUC@30), {mean_AUC_15_by_now:.4f} (AUC@15), {mean_AUC_5_by_now:.4f} (AUC@5), {mean_AUC_3_by_now:.4f} (AUC@3){RESET}")
        logger.info("="*80)

    # Print summary results
    logger.info("\nSummary of AUC results:")
    logger.info("-"*50)
    for category in sorted(per_category_results.keys()):
        logger.info(f"{category:<15}: {per_category_results[category]['Auc_30']:.4f} (AUC@30), {per_category_results[category]['Auc_15']:.4f} (AUC@15), {per_category_results[category]['Auc_5']:.4f} (AUC@5), {per_category_results[category]['Auc_3']:.4f} (AUC@3)")

    if per_category_results:
        mean_AUC_30 = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
        mean_AUC_15 = np.mean([per_category_results[category]["Auc_15"] for category in per_category_results])
        mean_AUC_5 = np.mean([per_category_results[category]["Auc_5"] for category in per_category_results])
        mean_AUC_3 = np.mean([per_category_results[category]["Auc_3"] for category in per_category_results])
        logger.info("-"*50)
        logger.info(f"Mean AUC: {mean_AUC_30:.4f} (AUC@30), {mean_AUC_15:.4f} (AUC@15), {mean_AUC_5:.4f} (AUC@5), {mean_AUC_3:.4f} (AUC@3)")

    total_s = __import__("time").perf_counter() - _eval_start
    if _seq_times:
        logger.info("="*50)
        logger.info(f"Performance summary ({len(_seq_times)} sequences, after warmup):")
        logger.info(f"  Model weights:    {weight_mem_mb} MB")
        _p50, _p90, _p99 = (float(x) for x in np.percentile(_seq_times, [50, 90, 99]))
        logger.info(f"  Per-sequence latency:  "
                    f"mean={np.mean(_seq_times):.1f}s  "
                    f"std={np.std(_seq_times):.1f}s  "
                    f"min={np.min(_seq_times):.1f}s  "
                    f"max={np.max(_seq_times):.1f}s")
        logger.info(f"  Latency percentiles:   "
                    f"p50={_p50:.1f}s  p90={_p90:.1f}s  p99={_p99:.1f}s")
        logger.info(f"  Total wall time: {total_s/60:.1f} min")
        logger.info(f"  Peak CUDA (last seq): allocated={torch.cuda.max_memory_allocated()//1024**2} MB  "
                    f"reserved={torch.cuda.max_memory_reserved()//1024**2} MB")

    # Save structured results to JSON for paper reporting.
    if per_category_results:
        _mean_30 = float(np.mean([v["Auc_30"] for v in per_category_results.values()]))
        _mean_15 = float(np.mean([v["Auc_15"] for v in per_category_results.values()]))
        _mean_5  = float(np.mean([v["Auc_5"]  for v in per_category_results.values()]))
        _mean_3  = float(np.mean([v["Auc_3"]  for v in per_category_results.values()]))

        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_path": model_path,
            "num_frames": num_frames,
            "weight_memory_mb": weight_mem_mb,
            "peak_cuda_mb": torch.cuda.max_memory_allocated() // 1024**2,
            "seq_latency_s": {
                "mean": float(np.mean(_seq_times)) if _seq_times else None,
                "std":  float(np.std(_seq_times))  if _seq_times else None,
                "min":  float(np.min(_seq_times))  if _seq_times else None,
                "max":  float(np.max(_seq_times))  if _seq_times else None,
                "p50":  float(np.percentile(_seq_times, 50)) if _seq_times else None,
                "p90":  float(np.percentile(_seq_times, 90)) if _seq_times else None,
                "p99":  float(np.percentile(_seq_times, 99)) if _seq_times else None,
                "n":    len(_seq_times),
            },
            "mean_auc": {
                "auc30": _mean_30,
                "auc15": _mean_15,
                "auc5":  _mean_5,
                "auc3":  _mean_3,
            },
            "per_category": {
                cat: {
                    "auc30": float(v["Auc_30"]),
                    "auc15": float(v["Auc_15"]),
                    "auc5":  float(v["Auc_5"]),
                    "auc3":  float(v["Auc_3"]),
                }
                for cat, v in per_category_results.items()
            },
        }

        out_path = results_path or "output/eval_results.json"
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with open(out_path, "w") as _f:
            json.dump(results, _f, indent=2)
        logger.info(f"Results saved to {out_path}")
    

def run_test_vggt(args):
    logger.info(f"==> Preparing testing for {args.dataset_name}..")

    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(args.world_size) if args.ddp else "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12396"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    with initialize(version_base=None, config_path="vggt/training/config"):
        cfg = compose(config_name="default")

    is_jetson = getattr(args, 'jetson', False)
    model_dtype = "float16" if is_jetson else "float32"
    logger.info(f"[{'Jetson' if is_jetson else 'Server'}] model_dtype={model_dtype}")
    if is_jetson:
        # NCCL probes P2P capability even with one GPU, which Jetson's NVML doesn't support.
        from omegaconf import OmegaConf
        cfg = OmegaConf.merge(cfg, {"distributed": {"backend": "gloo"}})
        logger.info("[Jetson] Switched distributed backend: NCCL → gloo (NVML P2P not supported)")
    trainer = Trainer(**cfg, model_dtype=model_dtype)
    trainer.model.module = replace(args, trainer.model.module)
    if args.init_from:
        trainer._load_resuming_checkpoint(args.init_from)
    chunk = 2 if is_jetson else 8
    logger.info(f"[{'Jetson' if is_jetson else 'Server'}] DPT frames_chunk_size={chunk}")
    run_evaluation_vggt(trainer.model.module, frames_chunk_size=chunk)

def replace(args, net):
    logger.info("==> Replacing model parameters..")
    replacement_dict = {
        nn.Conv2d : partial(Q.QConv2d, \
        num_bits=args.num_bits, w_grad_scale_mode = args.w_grad_scale_mode, \
        x_grad_scale_mode = args.x_grad_scale_mode, \
        weight_norm = args.weight_norm, w_quantizer = args.w_quantizer, x_quantizer = args.x_quantizer, \
        w_initializer = args.w_initializer, x_initializer = args.x_initializer), 
        nn.Linear: partial(Q.QLinear,  \
        num_bits=args.num_bits, w_grad_scale_mode = args.w_grad_scale_mode, \
        x_grad_scale_mode = args.x_grad_scale_mode, \
        weight_norm = args.weight_norm, w_quantizer = args.w_quantizer, x_quantizer = args.x_quantizer, \
        w_initializer = args.w_initializer, x_initializer = args.x_initializer)
    }
    exception_dict = {
        '__first__': partial(Q.QConv2d, num_bits=args.first_bits, w_quantizer =args.w_first_last_quantizer, x_quantizer = args.x_first_last_quantizer, \
                                w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, \
                                w_grad_scale_mode = args.w_first_last_grad_scale_mode, \
                                x_grad_scale_mode = args.x_first_last_grad_scale_mode, \
                                first_layer = True),
        '__camera_last__': partial(Q.QLinear,  num_bits=args.last_bits, w_quantizer =args.w_first_last_quantizer,x_quantizer = args.x_first_last_quantizer, \
                                w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, \
                                w_grad_scale_mode = args.w_first_last_grad_scale_mode, \
                                x_grad_scale_mode = args.x_first_last_grad_scale_mode, \
                                first_layer = False),
        '__depth_last__': partial(Q.QConv2d, num_bits=args.last_bits, w_quantizer =args.w_first_last_quantizer,x_quantizer = args.x_first_last_quantizer, \
                                w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, \
                                w_grad_scale_mode = args.w_first_last_grad_scale_mode, \
                                x_grad_scale_mode = args.x_first_last_grad_scale_mode, \
                                first_layer = False),
        '__last__': partial(Q.QLinear,  num_bits=args.last_bits, w_quantizer =args.w_first_last_quantizer,x_quantizer = args.x_first_last_quantizer, \
                                w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, \
                                w_grad_scale_mode = args.w_first_last_grad_scale_mode, \
                                x_grad_scale_mode = args.x_first_last_grad_scale_mode, \
                                first_layer = False),
    }
    net = replace_module(net, replacement_dict=replacement_dict, exception_dict=exception_dict, arch=args.model)
    logger.debug(f"Replaced model:\n {net}")
    return net

def load_ckp_vggt(args, trainer, load_scheduler=True):
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
    
    if "optimizer" in checkpoint and hasattr(trainer, 'optims'):
        logger.info("Loading optimizer state")
        if isinstance(trainer.optims, list):
            if isinstance(checkpoint["optimizer"], list):
                for i, optim in enumerate(trainer.optims):
                    optim.optimizer.load_state_dict(checkpoint["optimizer"][i])
            else:
                # Single optimizer in checkpoint but list expected
                trainer.optims[0].optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            trainer.optims.optimizer.load_state_dict(checkpoint["optimizer"])
    
    # Load scale optimizer if it exists
    if "s_optimizer" in checkpoint and hasattr(trainer, 's_optimizer') and trainer.s_optimizer is not None:
        logger.info("Loading scale optimizer state")
        trainer.s_optimizer.load_state_dict(checkpoint["s_optimizer"])
    
    # Get the epoch for scheduler loading
    loaded_epoch = checkpoint.get("prev_epoch", checkpoint.get("epoch", 0))
    
    if load_scheduler and "s_scheduler" in checkpoint and hasattr(trainer, 's_scheduler') and trainer.s_scheduler is not None:
        logger.info("Loading scale scheduler state")
        trainer.s_scheduler.load_state_dict(checkpoint["s_scheduler"], loaded_epoch)
    elif not load_scheduler:
        logger.info("Skipping scheduler state loading (each stage has its own scheduler)")
    
    if "prev_epoch" in checkpoint:
        trainer.epoch = checkpoint["prev_epoch"] + 1  # Resume from next epoch
        logger.info(f"Resuming from epoch {trainer.epoch}")
    elif "epoch" in checkpoint:
        # Fallback for older checkpoints
        trainer.epoch = checkpoint["epoch"]
        logger.info(f"Resuming from epoch {trainer.epoch}")
    
    if "steps" in checkpoint:
        trainer.steps = checkpoint["steps"]
        logger.info(f"Loaded steps: {trainer.steps}")
    
    if "time_elapsed" in checkpoint:
        trainer.ckpt_time_elapsed = checkpoint["time_elapsed"]
        logger.info(f"Loaded time elapsed: {trainer.ckpt_time_elapsed:.2f}s")

def run_train_vggt(rank, args):
    logger.info(f"==> Preparing training for {args.dataset_name}..")

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12397"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    with initialize(version_base=None, config_path="vggt/training/config"):
        cfg = compose(config_name="default")
    
    trainer = Trainer(**cfg)
    trainer.model.module = replace(args, trainer.model.module)
    trainer.gradient_clipper.setup_clipping(trainer.model)
    
    # run_load_model equivalent
    if args.action == 'load':
        stepsize_init(trainer.model.module, trainer.train_dataset.get_loader(0), args.device, num_batches=args.init_num, dataset_name=args.dataset_name)

    if args.init_from and args.action == 'resume':
        load_ckp_vggt(args, trainer)

    if args.different_optimizer_mode:
        sparams, params = split_params(\
            trainer.model.module, weight_decay=args.weight_decay, lr = args.lr, x_lr= args.x_step_size_lr, \
                w_lr= args.w_step_size_lr, x_wd = args.x_step_size_wd, w_wd = args.w_step_size_wd)
        soptimizer, sscheduler = scheduler_optimizer_class(args, sparams, args.step_size_optimizer)
        trainer.s_optimizer = soptimizer
        trainer.s_scheduler = sscheduler

    # Train mode
    trainer.model.module.train()
    torch.set_grad_enabled(True)

    trainer.run_train()