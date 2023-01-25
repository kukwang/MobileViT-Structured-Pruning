import os
import time
import random
import math
import collections
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from kd_loss import SoftTarget

from models import build_MobileVIT, get_model_config
from utils import *

def add_arguments(parser):
    dataset_list = ['cifar10', 'cifar100', 'imagenet']

    parser.add_argument('--seed', default=7, type=int, help='random seed (default: 7)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--device', default='cuda', help='device (default: cuda)')

    parser.add_argument('data', default='', metavar='DIR', help='path to dataset (default: None)')
    parser.add_argument('--dataset-name', default="cifar10", type=str, choices=dataset_list, help='name of the dataset (default: cifar10)')
    parser.add_argument('--classes', default=10, type=int, help='number of the class (default: 10)')
    parser.add_argument('--resize', default=64, type=int, help='resize size (default: 64)')

    parser.add_argument('--model-config', default='', help='model config file path (default: None)')
    parser.add_argument('--mode', default='s', help='select mode of the model (default: s)')
    parser.add_argument('--head_dim', type=int, help='dimension of the head')
    parser.add_argument('--head_num', type=int, help='number of the head')

    parser.add_argument('--test', action='store_true', help='test only mode (default: False)')
    parser.add_argument('--train-ratio', default=0.0, type=float, help='train/val split ratio (default: 0.0, not split))')
    parser.add_argument('--epoch', default=200, type=int, help='training epoch (default: 200)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--label-smoothing', default=0.1, type=float, help='label smoothing (default: 0.1)')
    parser.add_argument('--train-batch-size', default=128, type=int, help='batch size at training (default: 128)')
    parser.add_argument('--test-batch-size', default=1, type=int, help='batch size at inference (default: 1)')

    parser.add_argument('--kd-lambda', default=0.0, type=float, help='lambda in knowledge distillation (default: 0.0)')
    parser.add_argument('--kd-temp', default=4.0, type=float, help='temperature of softmax in knowledge distillation (default: 4.0)')

    parser.add_argument('--resume', default='', help='path of the model in training (default: None)')
    parser.add_argument('--dense-model', default='', help='path of the pruned model (default: None)')
    parser.add_argument('--pruned-model', default='', help='path of the pruned model (default: None)')
    parser.add_argument('--fprune-rate', default=0.29, type=float, help='pruning rate (filter, default: 0.29 (real pr:0.5))')

    parser.add_argument('--save-path', default=None, help='save path (default: None)')
    return parser

def main(args):
    
    # hold random seeds
    print(f"random seed: {args.seed}")
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed_all(args.seed + args.local_rank)
    np.random.seed(seed=args.seed + args.local_rank)
    random.seed(args.seed + args.local_rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # model save path
    if args.save_path is not None:
        save_path = args.save_path
    else:
        save_path = f'./save/mobilevit_{args.mode}_{args.dataset_name}_{args.epoch}ep_pr{args.fprune_rate}_finetuning.pth'

    # get datasets
    train_set, val_set, test_set = make_dataset(args)

    # build dataloader
    train_loader, val_loader, test_loader = make_dataloader(args, train_set, val_set, test_set)
    
    # build model
    model_config = get_model_config(args)

    if args.kd_lambda > 0.0:
        # load dense model
        dense_model = build_MobileVIT(args, model_config).to(args.device)
        print(f'{args.mode} size dense MobileViT parameter number:{count_parameters(dense_model)}')
        if args.dense_model:
            if os.path.isfile(args.dense_model):
                print(f"=> loading checkpoint '{args.dense_model}'")
                dense_model_configs = torch.load(args.dense_model)
                acc = dense_model_configs['top1_acc']
                dense_model.load_state_dict(dense_model_configs['state_dict'])
                print(f"=> loaded checkpoint '{args.dense_model}' (epoch {dense_model_configs['epoch']}) Top1_acc: {acc:f}")
            else:
                print(f"=> no checkpoint found at '{args.dense_model}'")

    # load pruned model
    pruned_model = build_MobileVIT(args, model_config, pr=True).to(args.device)
    print(f'{args.mode} size pr{args.fprune_rate} MobileViT parameter number:{count_parameters(pruned_model)}')
    if args.pruned_model:
        if os.path.isfile(args.pruned_model):
            print(f"=> loading pruned model '{args.pruned_model}'")
            pruned_model_config = torch.load(args.pruned_model)
            # masks = pruned_model_config['masks']
            pruned_model.load_state_dict(pruned_model_config['state_dict'])
            print(f"=> loaded pruned model '{args.pruned_model}' (sparsity {pruned_model_config['sparsity']})")
        else:
            print(f"=> no pruned model found at '{args.pruned_model}'")
    

    # set criterion and optimizer
    optimizer = optim.SGD(pruned_model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    if args.kd_lambda > 0:
        # Get loss function for KD
        kd_criterion = SoftTarget(args.kd_temp)

        print('Finetuning with KD start')
        if args.save_path is not None:
            save_path = args.save_path
        else:
            save_path = f'./save/mobilevit_{args.mode}_{args.dataset_name}_{args.epoch}ep_pr{args.fprune_rate}_finetuning_kd{args.kd_lambda}.pth'

        best_train_acc, avg_train_time = train(args, pruned_model, train_loader, val_loader, criterion, optimizer, scheduler, save_path,
                                               teacher_model=dense_model, kd_criterion=kd_criterion)

    else:
        print('Finetuning without KD start')
        best_train_acc, avg_train_time = train(args, pruned_model, train_loader, val_loader, criterion, optimizer, scheduler, save_path)

    print(f'Best val acc: {best_train_acc:.2f}%    Average training time: {avg_train_time:.2f}s')

    get_file_size(save_path)
    
    print('Test start')
    if os.path.isfile(save_path):
        print(f"=> loading checkpoint '{save_path}'")
        finetuned_model_configs = torch.load(save_path)
        top1_acc = finetuned_model_configs['top1_acc']
        pruned_model.load_state_dict(finetuned_model_configs['state_dict'])
        print(f"=> loaded checkpoint '{save_path}' (epoch {finetuned_model_configs['epoch']}) Acc: {top1_acc:f}")
    else:
        print(f"=> no checkpoint found at '{save_path}'")


    test_acc, test_time, avg_latency = test_latency(args, pruned_model, test_loader)
    print(f'Test acc: {test_acc:.2f}%    Test time: {test_time}s    Average test latency: {avg_latency:.2f}ms')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileViT Finetuning')
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)