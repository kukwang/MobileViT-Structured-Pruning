#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import time
from typing import Optional
import sys
import os

import torch
import torchvision
import torchvision.transforms as transforms

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_val_split(train_ratio, dataset):

    dataset_num = dataset.__len__()
    
    train_num = int(dataset_num * train_ratio)
    val_num = dataset_num - train_num
    train_set, val_set = torch.utils.data.random_split(dataset, [train_num, val_num])
    
    return train_set, val_set


def make_dataset(args):
    # make transforms
    train_transform = transforms.Compose([
                                transforms.Resize(args.resize),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)
                                ])

    test_transform = transforms.Compose([
                                transforms.Resize(args.resize),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)
                                ])

    if args.dataset_name == 'cifar10':
        # make datasets
        train_set = torchvision.datasets.CIFAR10(root=args.data,
                                               train=True,
                                               download=False,
                                               transform=train_transform)

        test_set = torchvision.datasets.CIFAR10(root=args.data,
                                               train=False,
                                               download=False,
                                               transform=test_transform)

    elif args.dataset_name == 'cifar100':
        # make datasets
        train_set = torchvision.datasets.CIFAR100(root=args.data,
                                                train=True,
                                                download=False,
                                                transform=train_transform)

        test_set = torchvision.datasets.CIFAR100(root=args.data,
                                                train=False,
                                                download=False,
                                                transform=test_transform)

    if args.train_ratio > 0.0:
        train_set, val_set = train_val_split(args.train_ratio, )
    else:
        val_set = None
           
    return train_set, val_set, test_set

def make_dataloader(args, train_set, test_set, val_set=None):
    # make dataloader
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.train_batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.test_batch_size,
                                              shuffle=False,
                                              drop_last=True,
                                              num_workers=2)

    if val_set is not None:
        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=args.train_batch_size,
                                                 shuffle=False,
                                                 drop_last=True,
                                                 num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    return train_loader, None, test_loader