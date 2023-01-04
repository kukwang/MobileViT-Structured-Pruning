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

def remain_num_after_pr(x, pr):
    return x - int(x*pr)

def convert_size(size_bytes):
    import math
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def get_file_size(path):
    file_size = convert_size(os.path.getsize(path)) 
    print(f'file size: {file_size} bytes')


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
        train_set, val_set = train_val_split(args.train_ratio, train_set)
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

def train(args, model, train_loader, val_loader, criterion, optimizer, save_path, teacher_model=None, kd_criterion=None):
    best_val_acc = 0
    total_train_time = 0
    
    # if args.kd_lambda > 0.0:
    #     print('Get soft labels')
    #     soft_label = []
    #     for _, data in enumerate(train_loader, 0):
    #         inputs, labels = data
    #         inputs, labels = inputs.to(args.device), labels.to(args.device)

    #         optimizer.zero_grad()

    #         teacher_outputs = teacher_model(inputs)
    #         soft_label.append(teacher_outputs)
        

    for epoch in range(args.epoch):
        # train
        train_total, train_correct, train_correct_teacher = 0, 0, 0
        train_loss = 0.0
        model.train()
        train_st = time.time()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            optimizer.zero_grad()

                
            outputs = model(inputs)
            normal_loss = criterion(outputs, labels)

            if args.kd_lambda > 0.0:
                # teacher_outputs = soft_label[i]
                teacher_outputs = teacher_model(inputs)
                kd_loss = kd_criterion(outputs, teacher_outputs) * args.kd_lambda
                loss = normal_loss + kd_loss

                _, teacher_pred = torch.max(teacher_outputs.data, 1)
                train_correct_teacher += (teacher_pred == labels).sum().item()
            else:
                loss = normal_loss

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
        val_total, val_correct, val_correct_teacher = 0, 0 ,0
        val_loss = 0.0
        model.eval()
        val_st = time.time()
        for _, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            outputs = model(inputs)
            normal_loss = criterion(outputs, labels)

            if args.kd_lambda > 0.0:
                teacher_outputs = teacher_model(inputs)
                kd_loss = kd_criterion(outputs, teacher_outputs) * args.kd_lambda
                loss = normal_loss + kd_loss

                _, teacher_pred = torch.max(teacher_outputs.data, 1)
                val_correct_teacher += (teacher_pred == labels).sum().item()
            else:
                loss = normal_loss

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_loss += loss.item()

        val_time = time.time() - val_st
        val_acc = val_correct / val_total * 100
        val_loss = val_loss / len(val_loader)
        print(f'epoch: {epoch + 1}    train_acc: {train_acc:.2f}%   train_loss: {train_loss:.2f}    train_time: {train_time:.2f}s    \
            val_acc: {val_acc:.2f}%    val_loss: {val_loss:.2f}    val_time: {val_time:.2f}s')
        if args.kd_lambda > 0:
            print(f'[teacher]    train acc: {train_correct_teacher / train_total * 100}  val acc:{val_correct_teacher / val_total * 100}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'top1_acc': val_acc},
                        save_path)

    return best_val_acc, total_train_time / args.epoch

def test(args, model, testloader):
    test_correct, test_total = 0, 0
    with torch.no_grad():
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