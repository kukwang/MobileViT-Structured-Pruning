# Filter Pruning
# https://proceedings.neurips.cc/paper/2020/file/ccb1d45fb76f7c5a0bf619f979c6cf36-Paper.pdf

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *

from models import build_MobileVIT, get_model_config, MobileViT
from utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    parser.add_argument('--fprune-rate', default=0.5, type=float, help='pruning rate (filter, default: 0.5)')

    parser.add_argument('--dense-model', default='', type=str, metavar='PATH', help='path to the model (default: None)')
    return parser

class Masking(object):
    def __init__(self, args, model: MobileViT):
        self.args = args                    # arguments
        self.modules = model.modules()      # modules in model

        self.masks = {}                     # masks of each layers
        # self.baseline_nonzero = 0           # count number of zeros in the original model
        self.name2zerofilters = {}          # number of pruned filters in specific layer
        # self.name2nonzerofilters = {}       # number of not pruned filters in specific layer
        self.name2zerofilters_idx = {}      # idxes of pruned filter in specific layer
        # self.name2nonzerofilters_idx = {}   # idxes of not pruned filter in specific layer
        
        print(f'init masks')
        self.mask_init(model)
        # for k in self.masks.keys():
        #     print(f'layer name: {k},    shape:{self.masks[k].shape}')
        
    def prune_filter(self, model: MobileViT):
        del_filter_idx_prev = torch.tensor([])  # pruned filters in previous conv layer
        print(f'make masks for pruning')
        # for module in model.modules():
        for name, weight in model.named_parameters():
            if name not in self.masks: continue
            if 'conv' in name:
                filter_abs_sum = torch.sum(weight.data.abs(), (1,2,3))
                num_filter_to_del = int(len(filter_abs_sum) * self.args.fprune_rate)
                _, del_filter_idx = torch.sort(filter_abs_sum)
                del_filter_idx = del_filter_idx[:num_filter_to_del]
            
                # prune columns (output channels)
                self.masks[name] = torch.ones_like(self.masks[name])
                self.masks[name][del_filter_idx.tolist()] = 0
                # print(f'masks:{self.masks[name]}')

                # save for remove filters
                self.name2zerofilters[name] = num_filter_to_del
                # self.name2nonzerofilters[name] = int(len(filter_abs_sum) - num_filter_to_del)
                self.name2zerofilters_idx[name] = set(del_filter_idx.tolist())
                # self.name2nonzerofilters_idx[name] = set(range(len(filter_abs_sum))) - self.name2zerofilters_idx[name]

                # prune rows (input channels)
                # ignore residual conenctions (it will be complemented by fine-tuning)
                if self.masks[name].shape[1] == 1:  # dwise conv
                    self.masks[name][del_filter_idx_prev.tolist()] = 0
                else:                               # normal conv
                    self.masks[name][:, del_filter_idx_prev.tolist()] = 0
                del_filter_idx_prev = del_filter_idx

            if 'net' in name:
                None

        print('-'*100)
        print(f'apply masks')
        model = self.apply_mask(model)

        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Sparsity after pruning: {0:.2f}%'.format(
            (total_size-sparse_size) / total_size * 100.))


    # ====================================================
    # pruning utils
    # ====================================================
    def mask_init(self, model: MobileViT):
        for name, weight in model.named_parameters():
            if weight.dim() > 1 and ('conv' in name or 'net' in name):      # conv2d layer, feedforward layer in transformer block
                self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).to(self.args.device)
                # self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()   
        
    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                # print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape, np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))


    def apply_mask(self, model: MobileViT):
        for name, weights in model.named_parameters():
            if name in self.masks:
                weights.data = weights.data * self.masks[name]
        return model


# ==================================================== main ====================================================

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

    mask_class = Masking(args, model)
    mask_class.prune_filter(model)

    # # 여기부터 수정
    # #{{{
    # # get the total shape of all BatchNorm layers
    # total = 0
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         total += m.weight.data.shape[0]

    # # get the |weight (gamma) of BatchNorm layers|
    # bn = torch.zeros(total)
    # index = 0
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         size = m.weight.data.shape[0]
    #         bn[index:(index+size)] = m.weight.data.abs().clone()
    #         index += size

    # # sort weight (gamma) of BatchNorm layers and get threshold point
    # y, i = torch.sort(bn)
    # threshold_index = int(total * args.fprune_rate)
    # threshold = y[threshold_index]

    # # make mask
    # pruned = 0
    # cfg = []        # saved parameters of each layers
    # cfg_mask = []   # mask of each layers
    # for k, m in enumerate(model.modules()):
    #     if isinstance(m, nn.BatchNorm2d):                       # every BatchNorm2d layer,
    #         weight_copy = m.weight.data.abs().clone()           # get absolute values of BatchNorm weight (gamma)
    #         mask = weight_copy.gt(threshold).float().cuda()     # (1.0 if weight > threshold else 0.0) for all elements in weight matrix
    #         pruned += mask.shape[0] - torch.sum(mask)           # save number of pruned weights
    #         m.weight.data.mul_(mask)                            # 
    #         m.bias.data.mul_(mask)
    #         cfg.append(int(torch.sum(mask)))
    #         cfg_mask.append(mask.clone())
    #         print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
    #             format(k, mask.shape[0], int(torch.sum(mask))))
    #     elif isinstance(m, nn.MaxPool2d):
    #         cfg.append('M')

    # pruned_ratio = pruned/total

    # print('Pre-processing Successful!')

    # print(f'pruned_ratio: {pruned_ratio}')

    # print("Cfg:")
    # print(cfg)

    # new_model = build_MobileVIT(args, model_config).to(args.device)
    # model.to(args.device)

    # num_parameters = sum([param.nelement() for param in new_model.parameters()])
    # save_path = "./prune.txt"
    # with open(save_path, "w") as fp:
    #     fp.write("Configuration: \n"+str(cfg)+"\n")
    #     fp.write("Number of parameters: \n"+str(num_parameters)+"\n")

    # # network slimming (prune) part
    # old_modules = list(model.modules())
    # new_modules = list(new_model.modules())
    # layer_id_in_cfg = 0
    # start_mask = torch.ones(3)              # init: start_mask = tensor([ 1.,  1.,  1.])
    # end_mask = cfg_mask[layer_id_in_cfg]    # init: end_mask = cfg_mask[0]
    # conv_count = 0

    # for layer_id in range(len(old_modules)):
    #     m0 = old_modules[layer_id]
    #     m1 = new_modules[layer_id]
    #     if isinstance(m0, nn.BatchNorm2d):
    #         idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
    #         if idx1.size == 1:
    #             idx1 = np.resize(idx1,(1,))

    #         m1.weight.data = m0.weight.data[idx1.tolist()].clone()
    #         m1.bias.data = m0.bias.data[idx1.tolist()].clone()
    #         m1.running_mean = m0.running_mean[idx1.tolist()].clone()
    #         m1.running_var = m0.running_var[idx1.tolist()].clone()
    #         layer_id_in_cfg += 1
    #         start_mask = end_mask.clone()
    #         if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
    #             end_mask = cfg_mask[layer_id_in_cfg]
    #     elif isinstance(m0, nn.Conv2d):
    #         if conv_count == 0:
    #             m1.weight.data = m0.weight.data.clone()
    #             conv_count += 1
    #             continue
    #         # if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
    #         if isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
    #             # This convers the convolutions in the residual block.
    #             # The convolutions are either after the channel selection layer or after the batch normalization layer.
    #             conv_count += 1
    #             idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
    #             idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
    #             print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
    #             if idx0.size == 1:
    #                 idx0 = np.resize(idx0, (1,))
    #             if idx1.size == 1:
    #                 idx1 = np.resize(idx1, (1,))
    #             w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

    #             # If the current convolution is not the last convolution in the residual block, then we can change the 
    #             # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
    #             if conv_count % 3 != 1:
    #                 w1 = w1[idx1.tolist(), :, :, :].clone()
    #             m1.weight.data = w1.clone()
    #             continue

    #         # We need to consider the case where there are downsampling convolutions. 
    #         # For these convolutions, we just copy the weights.
    #         m1.weight.data = m0.weight.data.clone()
    #     elif isinstance(m0, nn.Linear):
    #         idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
    #         if idx0.size == 1:
    #             idx0 = np.resize(idx0, (1,))

    #         m1.weight.data = m0.weight.data[:, idx0].clone()
    #         if m0.bias is not None:
    #             m1.bias.data = m0.bias.data.clone()

    # torch.save({'cfg': cfg, 'state_dict': new_model.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileViT pruning')
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)