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

    parser.add_argument('--dense-model', default='', type=str, metavar='PATH', help='path of the dense model (default: None)')
    parser.add_argument('--save-path', default='', type=str, metavar='PATH', help='path of the pruned model (default: None)')
    return parser

class Masking(object):
    def __init__(self, args, model: MobileViT, model_config):
        self.args = args                    # arguments
        self.modules = model.modules()      # modules in model
        self.model_config = model_config    # model config (expansion, dims, channels)

        self.masks = {}                     # masks of each layers
        # self.baseline_nonzero = 0           # count number of zeros in the original model
        self.name2zerofilters = {}          # number of pruned filters in specific layer
        self.name2nonzerofilters = {}       # number of not pruned filters in specific layer
        self.name2zerofilters_idx = {}      # idxes of pruned filter in specific layer
        self.name2nonzerofilters_idx = {}   # idxes of not pruned filter in specific layer
        
        print(f'init masks')
        self.mask_init(model)
        # for k in self.masks.keys():
        #     print(f'layer name: {k},    shape:{self.masks[k].shape}')
        
    def prune_filter(self, model: MobileViT):
        del_filter_idx_prev = torch.tensor([])      # pruned filters in previous conv layer
        del_filter_idx_prev_1x1 = torch.tensor([])  # pruned filters in prev 1x1 conv layer (for pruning concatenated weights in the MobileViTBlock)
        print(f'make masks for pruning')
        # for module in model.modules():
        for name, weight in model.named_parameters():
            if name not in self.masks: continue
            # conv2d layers
            if 'conv' in name:
                # dwise conv, prune same filter to the prev filters
                if self.masks[name].shape[1] == 1:
                    self.masks[name][del_filter_idx_prev.tolist()] = 0
                # normal conv
                else:
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
                    self.name2nonzerofilters[name] = int(len(filter_abs_sum) - num_filter_to_del)
                    self.name2zerofilters_idx[name] = set(del_filter_idx.tolist())
                    self.name2nonzerofilters_idx[name] = set(range(len(filter_abs_sum))) - self.name2zerofilters_idx[name]
                        
                    # prune rows (input channels)
                    # ignore residual conenctions (it will be complemented by fine-tuning)
                    self.masks[name][:, del_filter_idx_prev.tolist()] = 0
                    # conv layer after concatenate two weight tensors in MobileViTBlock
                    if 'conv4' in name:
                        del_filter_idx_prev_1x1 = [x + len(filter_abs_sum) for x in del_filter_idx_prev_1x1]    # shift the idxes
                        self.masks[name][:, del_filter_idx_prev_1x1] = 0
                    del_filter_idx_prev = del_filter_idx
                    if 'conv.6' in name:    # for pruning concatenated weights in the MobileViTBlock
                        del_filter_idx_prev_1x1 = del_filter_idx

            # linear layer in transformer feedforward
            elif 'net' in name:
                column_abs_sum = torch.sum(weight.data.abs(), 1)
                num_columns_to_del = int(len(column_abs_sum) * self.args.fprune_rate)
                _, del_column_idx = torch.sort(column_abs_sum)
                del_column_idx = del_column_idx[:num_columns_to_del]
            
                # prune columns (output channels)
                self.masks[name] = torch.ones_like(self.masks[name])
                self.masks[name][del_column_idx.tolist()] = 0
                # print(f'masks:{self.masks[name]}')

                # save for remove filters
                self.name2zerofilters[name] = num_columns_to_del
                self.name2nonzerofilters[name] = int(len(column_abs_sum) - num_columns_to_del)
                self.name2zerofilters_idx[name] = set(del_column_idx.tolist())
                self.name2nonzerofilters_idx[name] = set(range(len(column_abs_sum))) - self.name2zerofilters_idx[name]

                # prune rows (input channels)
                self.masks[name][:, del_filter_idx_prev.tolist()] = 0
                del_filter_idx_prev = del_column_idx

        print(f'apply masks')
        print('-'*100)
        zeroing_model = self.apply_mask(model)
        pruned_model = self.apply_mask_real(model)


        inp = torch.randn(128, 3, 256, 256).to(args.device)

        print(pruned_model(inp).size())

        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Sparsity after pruning: {0:.2f}%'.format(
            (total_size-sparse_size) / total_size * 100.))

        return pruned_model

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

    def apply_mask_real(self, model: MobileViT):
        pruned_weights = model.state_dict()
        batchnorm_weight = False
        prev_name, prev_1x1_name = None, None
        for k in pruned_weights.keys():
            if 'tracked' in k:  continue
            weights = pruned_weights[k]
            if batchnorm_weight:                    # batchnorm (weight, bias)
                weights = weights[list(self.name2nonzerofilters_idx[prev_name])]
                batchnorm_weight = False
            # bias in linear layer in feedforward, batchnorm, layernorm, linear in attention 
            elif ('bias' in k) or ('running' in k) or ('norm' in k) or ('to_out' in k):
                weights = weights[list(self.name2nonzerofilters_idx[prev_name])]
            # input to (query,key,value) in attention, fc layer 
            elif ('to_qkv' in k) or ('fc' in k):
                weights = weights[:,list(self.name2nonzerofilters_idx[prev_name])]
            elif ('conv' in k) or ('net' in k):     # conv2d layers, linear layer in transformer feedforward
                if self.masks[k].shape[1] == 1:     # dwise conv layer
                    weights = weights[list(self.name2nonzerofilters_idx[prev_name])]
                else:                               # normal conv
                    weights = weights[list(self.name2nonzerofilters_idx[k])]             # prune output channels
                    if prev_name is not None:
                        if 'conv4' in k:            # 
                            filter_idx = [x + len(weights)//2 for x in self.name2nonzerofilters_idx[prev_1x1_name]]    # shift the idxes
                            filter_idx = list(self.name2nonzerofilters_idx[k]) + filter_idx
                            weights = weights[:,filter_idx]
                        else:                       # normal weights
                            weights = weights[:,list(self.name2nonzerofilters_idx[prev_name])]  # prune input channels
                    if 'conv.6' in k:             # for pruning concatenated weights in the MobileViTBlock
                        prev_1x1_name = k

                    prev_name = k
                if 'conv' in k:  # to identify batchnorm
                    batchnorm_weight = True
            pruned_weights[k] = weights     # update weights

        pruned_model = build_MobileVIT(self.args, self.model_config, pr=True).to(args.device)
        
        pruned_model.load_state_dict(pruned_weights)    # 여기 모양 안맞다고 에러뜸
        return pruned_model

# ==================================================== main ====================================================

def main(args):
    # build model
    model_config = get_model_config(args)

    model = build_MobileVIT(args, model_config).to(args.device)
    model.to(args.device)

    # load dense model
    if args.dense_model:
        if os.path.isfile(args.dense_model):
            print(f"=> loading checkpoint '{args.dense_model}'")
            dense_model = torch.load(args.dense_model)
            acc = dense_model['top1_acc']
            model.load_state_dict(dense_model['state_dict'])
            print(f"=> loaded checkpoint '{args.dense_model}' (epoch {dense_model['epoch']}) Top1_acc: {acc:f}")
        else:
            print(f"=> no checkpoint found at '{args.dense_model}'")

    mask_class = Masking(args, model, model_config)
    pruned_model = mask_class.prune_filter(model)
    masks = mask_class.masks

    torch.save({'masks': masks,
                'state_dict': pruned_model.state_dict()},
                args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileViT Pruning')
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)