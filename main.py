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

from models import build_MobileVIT
from models import get_model_config
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

    parser.add_argument('--pruned-model', default='', help='path of pruned model (default: None)')

    return parser

def train(args, model, train_loader, val_loader, criterion, optimizer, save_path):
    best_val_acc = 0
    total_train_time = 0
    for epoch in range(args.epoch):
        # train
        train_total, train_correct = 0, 0
        train_loss = 0.0
        model.train()
        train_st = time.time()
        for _, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, pred = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_time = time.time() - train_st
        total_train_time += train_time
        train_acc = train_correct / train_total * 100
        train_loss = train_loss / len(train_loader) 
        
        # validation
        val_total, val_correct = 0, 0
        val_loss = 0.0
        model.eval()
        val_st = time.time()
        for _, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_loss += loss.item()

        val_time = time.time() - val_st
        val_acc = val_correct / val_total * 100
        val_loss = val_loss / len(val_loader)
        print(f'epoch: {epoch + 1}    train_acc: {train_acc:.2f}    train_loss: {train_loss:.2f}    train_time: {train_time:.2f}    \
            val_acc: {val_acc:.2f}    val_loss: {val_loss:.2f}    val_time: {val_time:.2f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    return best_val_acc, total_train_time // args.epoch

def test(args, model, testloader, save_path):
    test_correct, test_total = 0, 0
    with torch.no_grad():
        model.load_state_dict(torch.load(save_path))
        model.eval()
        test_st = time.time() 
        for _, data in enumerate(testloader):
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)

            outputs = model(images)

            _, pred = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (pred == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    test_time = time.time() - test_st 
    return test_acc, test_time

def main(args):
    
    # hold random seeds
    print(f"Using seed: {args.seed}")
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed_all(args.seed + args.local_rank)
    np.random.seed(seed=args.seed + args.local_rank)
    random.seed(args.seed + args.local_rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # model save path
    save_path = f'./save/mobilevit_{args.mode}_{args.dataset_name}_dense.pth'

    # get datasets
    # if train_ratio == 0.0, val_set is None
    train_set, val_set, test_set = make_dataset(args)

    # build dataloader
    # if train_ratio == 0.0, val_loader is None
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
        print(f'Best val acc: {best_train_acc:.2f} %    Average training time: {avg_train_time/args.epoch:.2f} s')

    print('Test start')
    test_acc, test_time = test(args, model, test_loader, save_path)
    print(f'Test acc: {test_acc:.2f} %    Test time: {test_time} s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileViT pruning')
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)