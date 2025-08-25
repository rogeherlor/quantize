import os
from src.logger import logger

import torch
import torch.nn as nn
from functools import partial

from hydra import initialize, compose
import src.models.depth.vggt.vggt
from src.models.depth.vggt.training.trainer import Trainer

from src.utils import replace_module, stepsize_init, split_params, Multiple_Loss, Multiple_optimizer_scheduler
from src.scheduler_optimizer_class import scheduler_optimizer_class
import src.module_quantization as Q

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
        '__last__': partial(Q.QLinear,  num_bits=args.last_bits, w_quantizer =args.w_first_last_quantizer,x_quantizer = args.x_first_last_quantizer, \
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

    if args.action == 'load':
        stepsize_init(trainer.model.module, trainer.train_dataset.get_loader(0), args.device, num_batches=1, dataset_name=args.dataset_name) #args.init_num
    
    # if args.different_optimizer_mode:
    #     sparams, params = split_params(\
    #         trainer.model.module, weight_decay=args.weight_decay, lr = args.lr, x_lr= args.x_step_size_lr, \
    #             w_lr= args.w_step_size_lr, x_wd = args.x_step_size_wd, w_wd = args.w_step_size_wd)
    # 
    # optimizer, scheduler = scheduler_optimizer_class(args, params, args.optimizer)
    # optimizer_dict = {"optimizer": optimizer}
    # scheduler_dict = {"scheduler": scheduler}
    # if args.different_optimizer_mode:
    #     soptimizer, sscheduler = scheduler_optimizer_class(args, sparams, args.step_size_optimizer)
    #     optimizer_dict["step_size_optimizer"] = soptimizer
    #     scheduler_dict["step_size_scheduler"] = sscheduler
    # 
    # task_loss_fn = nn.CrossEntropyLoss()
    # loss_dict = {"task_loss": task_loss_fn}
    # 
    # criterion = Multiple_Loss(loss_dict)
    # all_optimizers = Multiple_optimizer_scheduler(optimizer_dict)
    # all_schedulers = Multiple_optimizer_scheduler(scheduler_dict)
    
    # val_accuracy, val_top5_accuracy,  total_val_loss, best_acc, val_loss_dict = run_one_epoch(net, dataloader, all_optimizers, criterion, 0, "val", best_acc, args, ddp_initialized)
    trainer.model.module.train()
    torch.set_grad_enabled(True)

    # Modify training so scale is trained
    trainer.run_train()