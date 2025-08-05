from ..logger import logger
from .pytorchcv.vit import vitb16
from .pytorchcv.preresnet import preresnet18, preresnet34
from .pytorchcv.mobilenetv2 import mobilenetv2_w1

def create_model(args):
    model = None
    if args.dataset_name == 'imagenet' or args.dataset_name == 'imagenet-mini':
        if args.model == 'pytorchcv_preresnet18':
            model = preresnet18(pretrained=args.pre_trained)
        elif args.model == 'pytorchcv_preresnet34':
            model = preresnet34(pretrained=args.pre_trained)
        elif args.model == 'pytorchcv_mobilenetv2':
            model = mobilenetv2_w1(pretrained=args.pre_trained)
        elif args.model == 'pytorchcv_vitb16':
            model = vitb16(pretrained=args.pre_trained)
    elif args.dataset_name == 'RealEstate10K':
        if args.model == 'vggt':
            model = None
    if model == None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported', args.model, args.dataset_name)
        exit(-1)

    logger.info('Created `%s` model for `%s` dataset' % (args.model, args.dataset_name))
    logger.info('Use pre-trained model = %s' % args.pre_trained)

    return model

"""
def prepare_pretrained_model(args):
    if args.model == 'pytorchcv_preresnet18':
        args.init_from = 'model_zoo/pytorchcv/preresnet18-0972-5651bc2d.pth'
    elif args.model == 'pytorchcv_preresnet34':
        args.init_from = 'model_zoo/pytorchcv/preresnet34-0774-fd5bd1e8.pth'
    elif args.model == 'pytorchcv_mobilenetv2':
        args.init_from = 'model_zoo/pytorchcv/mobilenetv2_w1-0887-13a021bc.pth'
    elif args.model == 'pytorchcv_vitb16':
        args.init_from = 'model_zoo/pytorchcv/preresnet18-0972-5651bc2d.pth'
    elif args.model == 'vggt':
        args.init_from = 'model_zoo/pytorchcv/preresnet34-0774-fd5bd1e8.pth'
    logger.info("args.init_from", args.init_from)
"""