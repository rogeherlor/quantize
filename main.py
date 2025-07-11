#========================================================================
# import modules 
#========================================================================
import torch
import time
import yaml
import argparse

from src.logger import logger
from src.utils import *

from src.quantizer.uniform import *
from src.quantizer.nonuniform import *
from src.initializer import *
from src.run_qat import run_qat
from src.run_stats import run_stats

if __name__ == '__main__':
    with open('./config/imagenet/LSQ_base.yaml') as file:
        config = yaml.safe_load(file.read())
    config = dotdict(config)   
    config.world_size = torch.cuda.device_count()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Number of GPUs: {config.world_size}")
    logger.info(f"Device: {config.device}")

    parser = argparse.ArgumentParser(description='VGGT Quantization')
    parser.add_argument('--mode', type=str, default='QAT', help='mode: QAT or PTQ')
    parser.add_argument('--num_bits', type=int, help='quantization bit-width')
    parser.add_argument('--accelerator', type=str, default='GPU', help='accelerator: GPU or HAILO')
    # QAT parameters
    parser.add_argument('--lr', default=0.1, type=float, metavar='N', help='learning rate')
    parser.add_argument('--coeff_qparm_lr', default=0.1, type=float, metavar='N', help='qparm learning rate = coeff*lr')
    parser.add_argument('--weight_decay', default=0.1, type=float, metavar='N', help='weight decay')
    parser.add_argument('--qparm_wd', default=0.1, type=float, metavar='N', help='qparm_wd')
    parser.add_argument('--train_id', type=str, help='training id, is used for collect experiment results')
    parser.add_argument('--x_quantizer', type=str, help='x quantizer')
    parser.add_argument('--w_quantizer', type=str, help='w quantizer')
    parser.add_argument('--initializer', type=str, help='initializer')
    parser.add_argument('--first_run', action= 'store_true', help='control logs to reduce the redundancy')
    parser.add_argument('--init_from', type=str, help='init_from')

    # Config setup
    args = parser.parse_args()
    config.num_bits = args.num_bits
    if args.mode == 'QAT':
        QuantizerDict ={
            "MinMax_quantizer": MinMax_quantizer,
            "LSQ_quantizer": LSQ_quantizer,
            "LCQ_quantizer": LCQ_quantizer,
            "APoT_quantizer": APoT_quantizer,
            "Positive_nuLSQ_quantizer": Positive_nuLSQ_quantizer,
            "Symmetric_nuLSQ_quantizer": Symmetric_nuLSQ_quantizer,
            "FP": None
        }
        InitializerDict ={
            "NMSE_initializer": NMSE_initializer,
            "LSQ_initializer": LSQ_initializer,
            "Const_initializer": Const_initializer,
        }
        config.lr = args.lr
        config.x_step_size_lr = round(args.coeff_qparm_lr*config.lr, ndigits=5)
        config.w_step_size_lr = round(args.coeff_qparm_lr*config.lr, ndigits=5)
        config.weight_decay = args.weight_decay
        config.x_step_size_wd = args.qparm_wd
        config.w_step_size_wd = args.qparm_wd
        config.w_first_last_quantizer = MinMax_quantizer
        config.x_first_last_quantizer = MinMax_quantizer
        config.x_quantizer = QuantizerDict[args.x_quantizer]
        config.w_quantizer = QuantizerDict[args.w_quantizer]
        config.x_initializer = InitializerDict[args.initializer]
        config.w_initializer = InitializerDict[args.initializer]
        config.train_id = args.train_id
        config.first_run = args.first_run
        config.init_from = args.init_from if args.init_from != None else None

        if config.different_optimizer_mode == False:
            logger.info("reset Qparm hyper parameters to be same as other parameters' ones")
            config.step_size_optimizer = config.optimizer
            config.x_step_size_lr = config.lr
            config.w_step_size_lr = config.lr
            config.x_step_size_wd = config.weight_decay
            config.w_step_size_wd = config.weight_decay
    elif args.mode == 'PTQ':
        pass

    # Execution
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    if args.mode == 'QAT':
        logger.info("Start QAT at the following setting")
        logger.info(
            f"lr: {config.lr}, x_step_size_lr: {config.x_step_size_lr}, w_step_size_lr: {config.w_step_size_lr}, "
            f"weight_decay: {config.weight_decay}, x_step_size_wd: {config.x_step_size_wd}, w_step_size_wd: {config.w_step_size_wd}, "
            f"x_quantizer: {config.x_quantizer}, w_quantizer: {config.w_quantizer}, "
            f"x_initializer: {config.x_initializer}, w_initializer: {config.w_initializer}, "
            f"num_bits: {config.num_bits}, train_id: {config.train_id}"
        )
        run_qat(config)
        logger.info("QAT finished")
    elif args.mode == 'PTQ':
        logger.info("Start PTQ at the following setting")
        logger.warning("PTQ is not implemented yet")
    elif args.mode == 'STATS':
        logger.info("Start STATS at the following setting")
        run_stats(config)
        logger.info("STATS finished")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    logger.info(f"Total time: {end - start:.3f} seconds")