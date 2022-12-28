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
    
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--device', default='cuda', help='device')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--dataset-name', default="imagenet", type=str, choices=dataset_list, help='name of the dataset')
    parser.add_argument('--classes', default=1000, type=int, help='number of the class')
    parser.add_argument('--resize', default=256, type=int, help='resize size')

    parser.add_argument('--model-config', default=None, help='model config file path')
    parser.add_argument('--mode', default='s', help='select mode of the model')
    parser.add_argument('--head_dim', type=int, help='dimension of the head')
    parser.add_argument('--head_num', type=int, help='number of the head')

    parser.add_argument('--epoch', default=200, type=int, help='training epoch')
    parser.add_argument('--train-batch-size', default=64, type=int, help='batch size at training')
    parser.add_argument('--test-batch-size', default=1, type=int, help='batch size at inference')

    return parser

def train(args, model, trainloader, criterion, optimizer, save_path):
    best_acc = 0
    avg_time = 0
    for epoch in range(args.epoch):
        train_total, train_correct = 0, 0
        train_loss = 0.0
        ep_st = time.time()
        model.train()
        for _, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, pred = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        ep_time = time.time() - ep_st
        avg_time += ep_time
        train_acc = train_correct / train_total * 100
        train_loss = train_loss / len(trainloader) 
        
        print(f'epoch: {epoch + 1}    acc: {train_acc:.2f}    loss: {train_loss:.2f}    time: {ep_time:.2f}')
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), save_path)

    return best_acc, avg_time

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
    save_path = f'./save/mobilevit_{args.mode}.pth'

    # get datasets
    trainset, testset = make_dataset(args)

    # build dataloader
    trainloader, testloader = make_dataloader(args, trainset, testset)
    
    # build model
    model_config = get_model_config(args)

    model = build_MobileVIT(args, model_config).to(args.device)
    print(f'{args.mode} size mobileViT parameter number:{count_parameters(model)}')
    
    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print('Train start')
    best_train_acc, avg_train_time = train(args, model, trainloader, criterion, optimizer, save_path)
    print(f'Best training acc: {best_train_acc:.2f} %    Average training time: {avg_train_time/args.epoch:.2f} s')

    print('Test start')
    test_acc, test_time = test(args, model, testloader, save_path)
    print(f'Test acc: {test_acc:.2f} %    Test time: {test_time} s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileViT pruning')
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)