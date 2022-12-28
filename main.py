import os
import time
import random
import math
import collections
import argparse
import numpy as np

import torch
from models import build_MobileVIT
from models import get_model_config

def add_arguments(parser):
    parser.add_argument('--config', default=None, help='config file path')
    parser.add_argument('--classes', default=1000, type=int, help='number of the class')
    parser.add_argument('--mode', default='s', help='select mode of the model')
    parser.add_argument('--head_dim', type=int, help='dimension of the head')
    parser.add_argument('--head_num', type=int, help='number of the head')
    return parser

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    model_config = get_model_config(args)
    img = torch.randn(5, 3, 256, 256)

    vit = build_MobileVIT(args, model_config)
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileViT pruning')
    parser = add_arguments(parser)
    args = parser.parse_args()
    
    main(args)