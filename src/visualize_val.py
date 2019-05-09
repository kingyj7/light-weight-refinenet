"""
modified by yangjing from ./train.py
to visualize the valdation 
"""

# general libs
import argparse
import logging
import os
import random
import re
import sys
import time

# misc
import cv2
import numpy as np

# pytorch libs
import torch
import torch.nn as nn

# custom libs
from config import *
import pyximport
pyximport.install()
from miou_utils import compute_iu, fast_cm
from util import *

import matplotlib.pyplot as plt

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Full Pipeline Training")

    # Dataset
    parser.add_argument("--val-dir", type=str, default=VAL_DIR,
                        help="Path to the validation set directory.")
    parser.add_argument("--val-list", type=str, nargs='+', default=VAL_LIST,
                        help="Path to the validation set list.")
    parser.add_argument("--shorter-side", type=int, nargs='+', default=SHORTER_SIDE,
                        help="Shorter side transformation.")
    parser.add_argument("--crop-size", type=int, nargs='+', default=CROP_SIZE,
                        help="Crop size for training,")
    parser.add_argument("--normalise-params", type=list, default=NORMALISE_PARAMS,
                        help="Normalisation parameters [scale, mean, std],")
    parser.add_argument("--batch-size", type=int, nargs='+', default=BATCH_SIZE,
                        help="Batch size to train the segmenter model.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for pytorch's dataloader.")
    parser.add_argument("--num-classes", type=int, nargs='+', default=NUM_CLASSES,
                        help="Number of output classes for each task.")
    parser.add_argument("--low-scale", type=float, nargs='+', default=LOW_SCALE,
                        help="Lower bound for random scale")
    parser.add_argument("--high-scale", type=float, nargs='+', default=HIGH_SCALE,
                        help="Upper bound for random scale")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="Label to ignore during training")

    # Encoder
    parser.add_argument("--enc", type=str, default=ENC,
                        help="Encoder net type.")
    parser.add_argument("--enc-pretrained", type=bool, default=ENC_PRETRAINED,
                        help='Whether to init with imagenet weights.')
    # General
    parser.add_argument("--evaluate", type=bool, default=EVALUATE,
                        help='If true, only validate segmentation.')
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help='Seed to provide (near-)reproducibility.')
    parser.add_argument("--ckpt-path", type=str, default=CKPT_PATH,
                        help="Path to the checkpoint file.")
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY,
                        help='Print information every often.')

    return parser.parse_args()

def create_segmenter(
    net, pretrained, num_classes
    ):
    """Create Encoder; for now only ResNet [50,101,152]"""
    import sys;
    sys.path.append("../")
    from models.resnet import rf_lw50, rf_lw101, rf_lw152
    if str(net) == '50':
        return rf_lw50(num_classes, imagenet=pretrained)
    elif str(net) == '101':
        return rf_lw101(num_classes, imagenet=pretrained)
    elif str(net) == '152':
        return rf_lw152(num_classes, imagenet=pretrained)
    else:
        raise ValueError("{} is not supported".format(str(net)))

def create_loaders(
    val_dir, val_list,
    shorter_side, crop_size, low_scale, high_scale,
    normalise_params, batch_size, num_workers, ignore_label
    ):
    """
    """
    # Torch libraries
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    # Custom libraries
    from datasets import NYUDataset as Dataset
    from datasets import Pad, RandomCrop, RandomMirror, ResizeShorterScale, ToTensor, Normalise

    ## Transformations during training ##
    composed_trn = transforms.Compose([ResizeShorterScale(shorter_side, low_scale, high_scale),
                                    Pad(crop_size, [123.675, 116.28 , 103.53], ignore_label),
                                    RandomMirror(),
                                    RandomCrop(crop_size),
                                    Normalise(*normalise_params),
                                    ToTensor()])
    composed_val = transforms.Compose([Normalise(*normalise_params),
                                    ToTensor()])
    ## Training and validation sets ##

    valset = Dataset(data_file=val_list,
                         data_dir=val_dir,
                         transform_trn=None,
                         transform_val=composed_val)
    ## Training and validation loaders ##
    val_loader = DataLoader(valset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
    return val_loader

def load_ckpt(
    ckpt_path, ckpt_dict
    ):
    best_val = epoch_start = 0
    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(ckpt_path)
        for (k, v) in ckpt_dict.items():
            if k in ckpt:
                v.load_state_dict(ckpt[k])
        best_val = ckpt.get('best_val', 0)
        epoch_start = ckpt.get('epoch_start', 0)
        logger.info(" Found checkpoint at {} with best_val {:.4f} at epoch {}".
            format(
                ckpt_path, best_val, epoch_start
            ))
    return best_val, epoch_start


def validate(
    segmenter, val_loader, epoch, num_classes=-1
    ):
    """Validate segmenter
    """
    val_loader.dataset.set_stage('val')
    segmenter.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    plt.figure(figsize=(16, 12))
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            start = time.time()
            input = sample['image']  ##been transformed, if wanted to be shown, comment line 197
            target = sample['mask']
            input_var = torch.autograd.Variable(input).float().cuda()
            # Compute output
            output = segmenter(input_var)
            output = cv2.resize(output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                                target.size()[1:][::-1],
                                interpolation=cv2.INTER_CUBIC).argmax(axis=2).astype(np.uint8)
            
            #print(input[0].shape)      # 3*720*1280
            #print(target.shape)        # 1*720*1280

            # Compute IoU
            gt = target[0].data.cpu().numpy().astype(np.uint8)
            gt_idx = gt < num_classes # Ignore every class index larger than the number of classes
            cm += fast_cm(output[gt_idx], gt[gt_idx], num_classes)

            
            plt.subplot(1, 3, 1)
            plt.title('img-%d'%i)
            #inverse of 'image': (self.scale * image - self.mean) / self.std,
            unnormalized=(input[0].data.cpu().numpy().transpose(1, 2, 0)*args.normalise_params[2]+args.normalise_params[1])/args.normalise_params[0]
            plt.imshow(unnormalized.astype(np.uint8))            
            
            plt.subplot(1, 3, 2)
            plt.title('gt-%d'%i)
            plt.imshow(gt)
            
            plt.subplot(1, 3, 3)
            plt.title('pred-%d'%i)
            plt.imshow(output)
            #plt.show()                                                                              #in dataset.py comment line153
#           
            img_dir='/home/yangjing/code/wash-hand/light-weight-refinenet-master/infer_img/val_first10/40ep-77/'
            if i<10: 
                plt.savefig(os.path.join(img_dir,'%d.png'%i))

            if i % args.print_every == 0:
                logger.info(' Val epoch: {} [{}/{}]\t'
                            'Mean IoU: {:.3f}'.format(
                                epoch, i, len(val_loader),
                                compute_iu(cm).mean()
                            ))

    ious = compute_iu(cm)
    logger.info(" IoUs: {}".format(ious))
    miou = np.mean(ious)
    logger.info(' Val epoch: {}\tMean IoU: {:.3f}'.format(
                                epoch, miou))
    return miou

def main():
    global args, logger
    args = get_arguments()
    logger = logging.getLogger(__name__)
    ## Add args ##
    args.num_stages = len(args.num_classes)
    ## Set random seeds ##
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    ## Generate Segmenter ##
    segmenter = nn.DataParallel(
        create_segmenter(args.enc, args.enc_pretrained, args.num_classes[0])
        ).cuda()
    logger.info(" Loaded Segmenter {}, ImageNet-Pre-Trained={}, #PARAMS={:3.2f}M"
                .format(args.enc, args.enc_pretrained, compute_params(segmenter) / 1e6))
    ## Restore if any ##
    best_val, epoch_start = load_ckpt(args.ckpt_path, {'segmenter' : segmenter})

    for task_idx in range(args.num_stages):
        start = time.time()
        torch.cuda.empty_cache()
        ## Create dataloaders ##
        val_loader = create_loaders(
                                    args.val_dir,
                                    args.val_list[task_idx],
                                    args.shorter_side[task_idx],
                                    args.crop_size[task_idx],
                                    args.low_scale[task_idx],
                                    args.high_scale[task_idx],
                                    args.normalise_params,
                                    args.batch_size[task_idx],
                                    args.num_workers,
                                    args.ignore_label)
        if args.evaluate:
            return validate(segmenter, val_loader, 0, num_classes=args.num_classes[task_idx])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
