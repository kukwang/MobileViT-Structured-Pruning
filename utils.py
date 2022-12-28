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
        trainset = torchvision.datasets.CIFAR10(root=args.data,
                                               train=True,
                                               download=False,
                                               transform=train_transform)

        testset = torchvision.datasets.CIFAR10(root=args.data,
                                               train=False,
                                               download=False,
                                               transform=test_transform)

    elif args.dataset_name == 'cifar100':
        # make datasets
        trainset = torchvision.datasets.CIFAR100(root=args.data,
                                                train=True,
                                                download=False,
                                                transform=train_transform)

        testset = torchvision.datasets.CIFAR100(root=args.data,
                                                train=False,
                                                download=False,
                                                transform=test_transform)
        
    return trainset, testset

def make_dataloader(args, trainset, testset):
    # make dataloader
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=args.train_batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=2)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             drop_last=True,
                                             num_workers=2)
    return trainloader, testloader