# Network slimming
# https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *

from models import build_MobileVIT, get_model_config
from utils import *

def add_arguments(parser):
    parser.add_argument('--device', default='cuda', help='device (default: cuda)')

    parser.add_argument('--dataset-name', default='cifar10', type=str, help='training dataset (default: cifar10)')
    parser.add_argument('--classes', default=10, type=int, help='number of the class (default: 10)')
    parser.add_argument('--resize', default=64, type=int, help='resize size (default: 64)')

    parser.add_argument('--model-config', help='model config file path')
    parser.add_argument('--mode', default='s', type=str, help='select mode of the model (default: s)')
    parser.add_argument('--head_dim', type=int, help='dimension of the head')
    parser.add_argument('--head_num', type=int, help='number of the head')

    parser.add_argument('--test-batch-size', default=128, type=int, help='batch size at inference (default: 128)')
    parser.add_argument('--cprune-rate', default=0.5, type=float, help='pruning rate (in channel, default: 0.5)')

    parser.add_argument('--dense-model', default='', type=str, metavar='PATH', help='path to the model (default: None)')

    return parser

def main(args):
    # build model
    model_config = get_model_config(args)

    model = build_MobileVIT(args, model_config).to(args.device)
    model.to(args.device)

    # load trained dense model
    if args.dense_model:
        if os.path.isfile(args.dense_model):
            print(f"=> loading checkpoint '{args.dense_model}'")
            dense_model = torch.load(args.dense_model)
            acc = dense_model['top1_acc']
            model.load_state_dict(dense_model['state_dict'])
            print(f"=> loaded checkpoint '{args.dense_model}' (epoch {dense_model['epoch']}) Top1_acc: {acc:f}")
        else:
            print(f"=> no checkpoint found at '{args.dense_model}'")

    # get the total shape of all BatchNorm layers
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    # get the |weight of BatchNorm layers|
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    # sort weight of BatchNorm layers and get threshold point
    y, i = torch.sort(bn)
    threshold_index = int(total * args.cprune_rate)
    threshold = y[threshold_index]


    # make mask
    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(threshold).float().cuda()
            pruned += mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned/total

    print('Pre-processing Successful!')

    print(f'pruned_ratio: {pruned_ratio}')

    print("Cfg:")
    print(cfg)

    new_model = build_MobileVIT(args, model_config).to(args.device)
    model.to(args.device)

    num_parameters = sum([param.nelement() for param in new_model.parameters()])
    save_path = os.path.join(args.save, "prune.txt")
    with open(save_path, "w") as fp:
        fp.write("Configuration: \n"+str(cfg)+"\n")
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")

    # network slimming (prune) part
    old_modules = list(model.modules())
    new_modules = list(new_model.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)              # init: start_mask = tensor([ 1.,  1.,  1.])
    end_mask = cfg_mask[layer_id_in_cfg]    # init: end_mask = cfg_mask[0]
    conv_count = 0

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            # if isinstance(old_modules[layer_id + 1], channel_selection):
            #     # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
            #     m1.weight.data = m0.weight.data.clone()
            #     m1.bias.data = m0.bias.data.clone()
            #     m1.running_mean = m0.running_mean.clone()
            #     m1.running_var = m0.running_var.clone()

            #     # We need to set the channel selection layer.
            #     m2 = new_modules[layer_id + 1]
            #     m2.indexes.data.zero_()
            #     m2.indexes.data[idx1.tolist()] = 1.0

            #     layer_id_in_cfg += 1
            #     start_mask = end_mask.clone()
            #     if layer_id_in_cfg < len(cfg_mask):
            #         end_mask = cfg_mask[layer_id_in_cfg]
            # else:
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            # if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
            if isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the 
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                if conv_count % 3 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions. 
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    torch.save({'cfg': cfg, 'state_dict': new_model.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileViT pruning')
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)