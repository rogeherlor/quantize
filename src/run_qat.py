import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter

from hydra import initialize, compose

from functools import partial

from src.logger import logger
from src.utils import setup_dataloader, replace_module

from src.models.model import create_model

import src.module_quantization as Q

from src.utils import *
from src.scheduler_optimizer_class import scheduler_optimizer_class
from src.make_ex_name import *

def run_test(args):
    logger.info(f"==> Preparing testing for {args.dataset_name}..")

    if args.model == 'vggt':
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(args.world_size) if args.ddp else "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12395"
        os.environ["NCCL_P2P_DISABLE"] = "1"

        with initialize(version_base=None, config_path="models/depth/vggt/training/config"):
            cfg = compose(config_name="default")

        trainer = Trainer(**cfg)
        trainer.run_val()
    else:
        pin_memory = True if args.device == 'cuda' else False
        dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory = pin_memory, DDP_mode = False, model = args.model)
        net = create_model(args)
        assert net != None
        
        net = net.to(args.device)

        if args.init_from and os.path.isfile(args.init_from):
            logger.info('==> Loading from checkpoint: ', args.init_from)
            net = run_load_model(args)
            # net = load_from_FP32_model(args.init_from, net)

        cudnn.benchmark = True if args.device == 'cuda' else False

        task_loss_fn = nn.CrossEntropyLoss()
        criterion_val = Multiple_Loss( {"task_loss": task_loss_fn})       
            
        val_accuracy, val_top5_accuracy,  val_loss, best_acc, val_loss_dict   = run_one_epoch(net, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False)
        logger.info(f'[FP32 model] val_Loss: {val_loss_dict["task_loss"].item():.5f}, val_top1_Acc: {val_accuracy:.5f}, val_top5_Acc: {val_top5_accuracy:.5f}')
        
        if args.write_log:
            args.ex_name = make_ex_name(args)
            save_path = os.path.join(args.save_path, "test", args.dataset_name, args.ex_name)
            os.makedirs(save_path, exist_ok=True)
            writer = SummaryWriter(save_path)
            writer.add_scalar("top1_accuracy/2.val", val_accuracy, 0)
            writer.add_scalar("top5_accuracy/2.val", val_top5_accuracy, 0)
            writer.add_scalar("loss/3.val_cross_entropy", val_loss_dict["task_loss"], 0)
            log_detailed_params(writer, net, prefix='test-')
    
def run_load_model(args):
    pin_memory = True if args.device == 'cuda' else False
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory = pin_memory, DDP_mode = False, model = args.model)
    net = create_model(args)
    net = net.to(args.device)
    assert net != None

    if not args.ddp:
        logger.info("Saving original model state dict..")
        torch.save(net.state_dict(), f"./model_zoo/pytorchcv/{args.model}_{args.dataset_name}_original.pth")

    # logger.info("Exporting original model to ONNX format..")
    # onnx_dataloader = setup_dataloader(args.dataset_name, args.batch_size, nworkers=0, pin_memory = False, DDP_mode=False, model=args.model)
    # dummy_input, _ = next(iter(onnx_dataloader["train"]))
    # dummy_input = dummy_input.to(args.device)
    # torch.onnx.export(net, dummy_input, f"./model_zoo/onnx/{args.model}_{args.dataset_name}_original.onnx")

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
        '__last_for_mobilenet__': partial(Q.QConv2d, num_bits=args.last_bits, w_quantizer =args.w_first_last_quantizer, x_quantizer = args.x_first_last_quantizer, \
                                w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, 
                                w_grad_scale_mode = args.w_first_last_grad_scale_mode, \
                                x_grad_scale_mode = args.x_first_last_grad_scale_mode, \
                                first_layer = False),
    }

    net = replace_module(net, replacement_dict=replacement_dict, exception_dict=exception_dict, arch=args.model)
    logger.debug(net)

    logger.info(f"==> Performing action: {args.action}")
    if args.action == 'load':
        if args.init_from and os.path.isfile(args.init_from):
            logger.info(f'==> Initializing from checkpoint: {args.init_from}')
            net = load_from_FP32_model(args.init_from, net)
            net.to(args.device)
            stepsize_init(net, dataloader["train"], args.device, args.init_num)
        else:
            logger.warning(f"No valid checkpoint file is provided !!! {args.init_from}")
    
    net = net.to(args.device)
    return net

def run_one_epoch(net, dataloader, optimizers, criterion, epoch, mode, best_acc, args, ddp_initialized=False):
    if mode == "train":
        net.train()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        torch.set_grad_enabled(False)

    total_loss = 0.0
    total_loss_dict = {}
    total_num_sample  = 0
    total_num_correct = 0
    total_num_correct5 = 0
    
    with tqdm(total=len(dataloader[mode]), disable = args.invisible_pgb) as pbar:
        pbar.set_description(f"Epoch[{epoch}/{args.nepochs}]({mode})")

        # loop for each epoch
        for i, (inputs, labels) in enumerate(dataloader[mode]):
            loss_dict = {}
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            
            # run train/val/test
            if mode == "train":
                optimizers.zero_grad()
                outputs = net(inputs)
                loss, loss_dict = criterion(outputs, labels)
                loss.backward()
                optimizers.step()
            else:
                # outputs = net(inputs).cuda() if args.device == 'cuda' else net(inputs)
                with torch.no_grad():
                    outputs = net(inputs)
                    loss, loss_dict = criterion(outputs, labels)

            # statistics
            _, predicted       = torch.max(outputs.detach(), 1)
            _, predicted5       = outputs.topk(5, 1, True, True)
            predicted5 = predicted5.t()
            num_sample         = labels.size(0)
            num_correct        = int((predicted == labels).sum())  
            targets = labels.expand_as(predicted5)
            num_correct5       = predicted5.eq(targets).reshape(-1).float().sum(0)
            total_loss        += loss.detach() * num_sample
            total_loss_dict = {k: total_loss_dict.get(k, 0) + loss_dict.get(k, 0).detach()* num_sample for k in set(total_loss_dict) | set(loss_dict)}
            total_num_sample  += num_sample
            total_num_correct += num_correct
            total_num_correct5 += num_correct5
            
            pbar.update(1)

    if ddp_initialized and args.ddp:
        total_loss, total_num_correct, total_num_correct5 = parallel_reduce(total_loss, total_num_correct, total_num_correct5)
        total_loss_dict = parallel_reduce_for_dict(total_loss_dict)
        total_num_sample = args.world_size * total_num_sample
    avg_loss = total_loss / total_num_sample
    avg_loss_dict = {k: total_loss_dict.get(k, 0)/total_num_sample for k in set(total_loss_dict)}
    accuracy = total_num_correct / total_num_sample
    top5_accuracy = total_num_correct5 / total_num_sample
    if accuracy > best_acc and mode != 'train':
        best_acc = accuracy
        
    return accuracy, top5_accuracy, avg_loss, best_acc, avg_loss_dict

def run_train(rank, args):
    start_epoch = 0
    world_size = args.world_size

    if args.ddp:
        args.device = f'cuda:{rank}'
        ddp_setup(rank, world_size)
        data_mode = True
        ddp_initialized = torch.distributed.is_initialized()
    else:
        data_mode = False
        ddp_initialized = False

    best_acc = 0.0

    logger.info(f"==> Preparing training for {args.model} {args.dataset_name}..")
    pin_memory = True if args.device == 'cuda' else False
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory = pin_memory, DDP_mode = data_mode, model = args.model)
    
    net = run_load_model(args)

    if args.write_log:
        args.ex_name = make_ex_name(args)
        save_path = os.path.join(args.save_path, "train", args.dataset_name, args.ex_name)
        os.makedirs(save_path, exist_ok=True)
        write_experiment_params(args, os.path.join(save_path, args.model+'.txt'))
        writer = SummaryWriter(save_path)

        if rank == 0 or not args.ddp:
            if args.first_run:
                logger.debug(net) 
                logger.info(f"Number of learnable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6:.2f} M")

    if torch.cuda.device_count() >= 1 and ddp_initialized:
        if rank == 0 or not args.ddp:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        # Convert BatchNorm to SyncBatchNorm. 
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net, device_ids=[rank])
        cudnn.benchmark = True
    
    # split parameters for different optimizers
    if args.different_optimizer_mode:
        sparams, params = split_params(\
            net, weight_decay=args.weight_decay, lr = args.lr, x_lr= args.x_step_size_lr, \
                w_lr= args.w_step_size_lr, x_wd = args.x_step_size_wd, w_wd = args.w_step_size_wd)
        if rank == 0 or not args.ddp:
            logger.info(f"sparams {sparams}")
            logger.info("========================================================================================================================")
    else:
        params = net.parameters()

    # setup optimizer & scheduler        
    optimizer, scheduler = scheduler_optimizer_class(args, params, args.optimizer)
    optimizer_dict = {"optimizer": optimizer}
    scheduler_dict = {"scheduler": scheduler}
    if args.different_optimizer_mode:
        soptimizer, sscheduler = scheduler_optimizer_class(args, sparams, args.step_size_optimizer)
        optimizer_dict["step_size_optimizer"] = soptimizer
        scheduler_dict["step_size_scheduler"] = sscheduler

    if args.action == 'resume':
        if args.init_from and os.path.isfile(args.init_from):
            start_epoch, net, optimizer, scheduler, _, acc= load_ckp(args.init_from, net, optimizer, scheduler)
            logger.info("acc=", acc)
            best_acc = acc.to(rank)
            logger.info("best_acc=", best_acc)
        else:
            logger.warning("No checkpoint file is provided !!!")

    # Loss
    task_loss_fn = nn.CrossEntropyLoss()
    loss_dict = {"task_loss": task_loss_fn}

    criterion = Multiple_Loss(loss_dict)
    all_optimizers = Multiple_optimizer_scheduler(optimizer_dict)
    all_schedulers = Multiple_optimizer_scheduler(scheduler_dict)

    logger.info("Inference before training..")
    val_accuracy, val_top5_accuracy,  total_val_loss, best_acc, val_loss_dict = run_one_epoch(net, dataloader, all_optimizers, criterion, 0, "val", best_acc, args, ddp_initialized)
    if rank == 0 or not args.ddp:
        logger.info(f'Before learning val_Loss: {val_loss_dict["task_loss"].item():.4f}, val_Acc: {val_accuracy:.4f}')
    logger.info("Training..")
    for epoch in range(start_epoch, args.nepochs):
        if args.ddp:
            dataloader["train"].sampler.set_epoch(epoch)
        train_accuracy, train_top5_accuracy, total_train_loss, _, train_loss_dict = run_one_epoch(net, dataloader, all_optimizers, criterion, epoch, "train", best_acc, args, ddp_initialized)
        val_accuracy, val_top5_accuracy,  total_val_loss, best_acc, val_loss_dict   = run_one_epoch(net, dataloader, all_optimizers, criterion, epoch, "val", best_acc, args, ddp_initialized)
        if rank == 0 or not args.ddp:
            logger.info(f"[Train] Epoch= {epoch}, train_total_loss: {total_train_loss.item():.5f}, train_task_loss: {train_loss_dict['task_loss'].item():.5f}, train_top1_Acc: {train_accuracy:.5f}, train_top5_Acc: {train_top5_accuracy:.5f}")
            logger.info(f"[Val] Epoch= {epoch}, val_Loss: {val_loss_dict['task_loss'].item():.5f}, val_top1_Acc: {val_accuracy:.3f}, val_top5_Acc: {val_top5_accuracy:.5f}")
        # update coefficients by scheduler
        all_schedulers.step()

        if args.write_log:
            writer.add_scalar("loss/1.train_total", total_train_loss, epoch)
            writer.add_scalar("loss/2.train_cross_entropy", train_loss_dict["task_loss"], epoch)
            writer.add_scalar("loss/3.val_cross_entropy", val_loss_dict["task_loss"], epoch)
            writer.add_scalar("top1_accuracy/1.train", train_accuracy, epoch)
            writer.add_scalar("top1_accuracy/2.val", val_accuracy, epoch)
            writer.add_scalar("top5_accuracy/1.train", train_top5_accuracy, epoch)
            writer.add_scalar("top5_accuracy/2.val", val_top5_accuracy, epoch)
            log_detailed_params(writer, net)

            if args.save_mode == "all_checkpoints":
                save_ckp(net, scheduler, optimizer, best_acc, epoch, val_accuracy, args.ddp, filename=os.path.join(save_path, 'ckpt_{}epoch.pth'.format(epoch)))
            else:
                save_ckp(net, scheduler, optimizer, best_acc, epoch, val_accuracy, args.ddp, filename=os.path.join(save_path, 'ckpt.pth'))
            if best_acc == val_accuracy:
                logger.info('Saving best acc model ..')
                save_ckp(net, None, None, best_acc, epoch, best_acc, args.ddp, filename=os.path.join(save_path, 'best.pth'))
    
    if ddp_initialized:
        torch.distributed.barrier()
        destroy_process_group()

def run_qat(args):
    logger.debug("Run QAT arguments:\n", args)

    if args.evaluation_mode:
        args.ddp = False
        logger.info(f"Visible GPU rank: {args.gpu_rank}. Default cuda:0")
        run_test(args)
    else:
        if args.ddp == True:
            logger.info("DDP mode")
            mp.spawn( \
                run_train, \
                nprocs= args.world_size, \
                args= (args,) \
            )
        else:
            logger.info("Non-DDP mode")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            logger.info(f"Visible GPU rank: {args.gpu_rank}. Default cuda:0")
            run_train(0, args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            logger.info(f"Training elapsed time: {end - start:.3f} seconds")