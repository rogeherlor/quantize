import os
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

def run_evaluation_vggt(model, model_path=None):
    """Main function to evaluate VGGT on CO3D dataset. Original vggt/evaluation/test_co3d.py"""
    args = argparse.Namespace()
    args.debug = False
    args.use_ba = False
    args.fast_eval = False
    args.min_num_images = 50
    args.num_frames = 10
    args.co3d_dir = "data3/rogelio/co3d/dataset/"
    args.co3d_anno_dir = "data3/rogelio/co3d/preprocessed_dataset/"
    args.seed = 0
    args.model_path = model_path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Set random seeds
    set_random_seeds(args.seed)

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

    per_category_results = {}

    for category in SEEN_CATEGORIES:
        print(f"Loading annotation for {category} test set")
        annotation_file = os.path.join(args.co3d_anno_dir, f"{category}_test.jgz")

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        rError = []
        tError = []

        seq_names = sorted(list(annotation.keys()))
        if args.fast_eval and len(seq_names)>=10:
            seq_names = random.sample(seq_names, 10)
        seq_names = sorted(seq_names)


        print("Testing Sequences: ")
        print(seq_names)

        for seq_name in seq_names:
            seq_data = annotation[seq_name]
            print("-" * 50)
            print(f"Processing {seq_name} for {category} test set")
            if args.debug and not os.path.exists(os.path.join(args.co3d_dir, category, seq_name)):
                print(f"Skipping {seq_name} (not found)")
                continue

            seq_rError, seq_tError = process_sequence(
                model, seq_name, seq_data, category, args.co3d_dir,
                args.min_num_images, args.num_frames, args.use_ba, device, dtype,
            )

            print("-" * 50)

            if seq_rError is not None and seq_tError is not None:
                rError.extend(seq_rError)
                tError.extend(seq_tError)

        if not rError:
            print(f"No valid sequences found for {category}, skipping")
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

        print("="*80)
        # Print results with colors
        GREEN = "\033[92m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        print(f"{BOLD}{BLUE}AUC of {category} test set:{RESET} {GREEN}{Auc_30:.4f} (AUC@30), {Auc_15:.4f} (AUC@15), {Auc_5:.4f} (AUC@5), {Auc_3:.4f} (AUC@3){RESET}")
        mean_AUC_30_by_now = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
        mean_AUC_15_by_now = np.mean([per_category_results[category]["Auc_15"] for category in per_category_results])
        mean_AUC_5_by_now = np.mean([per_category_results[category]["Auc_5"] for category in per_category_results])
        mean_AUC_3_by_now = np.mean([per_category_results[category]["Auc_3"] for category in per_category_results])
        print(f"{BOLD}{BLUE}Mean AUC of categories by now:{RESET} {RED}{mean_AUC_30_by_now:.4f} (AUC@30), {mean_AUC_15_by_now:.4f} (AUC@15), {mean_AUC_5_by_now:.4f} (AUC@5), {mean_AUC_3_by_now:.4f} (AUC@3){RESET}")
        print("="*80)

    # Print summary results
    print("\nSummary of AUC results:")
    print("-"*50)
    for category in sorted(per_category_results.keys()):
        print(f"{category:<15}: {per_category_results[category]['Auc_30']:.4f} (AUC@30), {per_category_results[category]['Auc_15']:.4f} (AUC@15), {per_category_results[category]['Auc_5']:.4f} (AUC@5), {per_category_results[category]['Auc_3']:.4f} (AUC@3)")

    if per_category_results:
        mean_AUC_30 = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
        mean_AUC_15 = np.mean([per_category_results[category]["Auc_15"] for category in per_category_results])
        mean_AUC_5 = np.mean([per_category_results[category]["Auc_5"] for category in per_category_results])
        mean_AUC_3 = np.mean([per_category_results[category]["Auc_3"] for category in per_category_results])
        print("-"*50)
        print(f"Mean AUC: {mean_AUC_30:.4f} (AUC@30), {mean_AUC_15:.4f} (AUC@15), {mean_AUC_5:.4f} (AUC@5), {mean_AUC_3:.4f} (AUC@3)")
    print(args.model_path)

def run_test_vggt(args):
    logger.info(f"==> Preparing testing for {args.dataset_name}..")

    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(args.world_size) if args.ddp else "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12395"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    with initialize(version_base=None, config_path="vggt/training/config"):
        cfg = compose(config_name="default")

    trainer = Trainer(**cfg)
    trainer.run_val()

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
    }
    net = replace_module(net, replacement_dict=replacement_dict, exception_dict=exception_dict, arch=args.model)
    logger.debug(f"Replaced model:\n {net}")
    return net

def run_train_vggt(rank, args):
    logger.info(f"==> Preparing training for {args.dataset_name}..")

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12395"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    
    with initialize(version_base=None, config_path="vggt/training/config"):
        cfg = compose(config_name="default")
    
    trainer = Trainer(**cfg)
    trainer.model.module = replace(args, trainer.model.module)
    
    # run_load_model equivalent
    if args.action == 'load':
        stepsize_init(trainer.model.module, trainer.train_dataset.get_loader(0), args.device, num_batches=args.init_num, dataset_name=args.dataset_name)

    if args.init_from and args.action == 'resume':
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
            trainer.optims.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Load scale optimizer if it exists
        if "s_optimizer" in checkpoint and hasattr(trainer, 's_optimizer') and trainer.s_optimizer is not None:
            logger.info("Loading scale optimizer state")
            trainer.s_optimizer.load_state_dict(checkpoint["s_optimizer"])
        
        if "s_scheduler" in checkpoint and hasattr(trainer, 's_scheduler') and trainer.s_scheduler is not None:
            logger.info("Loading scale scheduler state")
            trainer.s_scheduler.load_state_dict(checkpoint["s_scheduler"])
        
        if "epoch" in checkpoint:
            trainer.epoch = checkpoint["epoch"]
        if "steps" in checkpoint:
            trainer.steps = checkpoint["steps"]
        if "time_elapsed" in checkpoint:
            trainer.ckpt_time_elapsed = checkpoint["time_elapsed"]

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