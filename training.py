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
    save_path = f'./save/mobilevit_{args.mode}_{args.dataset_name}_{args.epoch}ep_dense.pth'

    # get datasets
    train_set, val_set, test_set = make_dataset(args)

    # build dataloader
    train_loader, val_loader, test_loader = make_dataloader(args, train_set, val_set, test_set)
    
    # build model
    model_config = get_model_config(args)

    model = build_MobileVIT(args, model_config).to(args.device)
    print(f'{args.mode} size MobileViT parameter number:{count_parameters(model)}')
    
    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if not args.test:
        print('Train start')
        best_train_acc, avg_train_time = train(args, model, train_loader, val_loader, criterion, optimizer, save_path)
        print(f'Best val acc: {best_train_acc:.2f}%    Average training time: {avg_train_time:.2f}s')

    print(f'model size: {get_file_size(save_path)}')
    
    print('Test start')
    if os.path.isfile(save_path):
        print(f"=> loading checkpoint '{save_path}'")
        dense_model = torch.load(save_path)
        top1_acc = dense_model['top1_acc']
        model.load_state_dict(dense_model['state_dict'])
        print(f"=> loaded checkpoint '{save_path}' (epoch {dense_model['epoch']}) Acc: {top1_acc:f}")
    else:
        print(f"=> no checkpoint found at '{save_path}'")


    test_acc, test_time = test(args, model, test_loader)
    print(f'Test acc: {test_acc:.2f}%    Test time: {test_time}s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileViT Training')
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)