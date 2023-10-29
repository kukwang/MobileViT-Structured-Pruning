#  Compressing MobileViT using Structured Pruning

## Introduction

Apply filter pruning to MobileViT specified in ["MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"](https://arxiv.org/abs/2110.02178).

## Usage

#### Pruning
```
python3 filter_prune.py /path/to/datset --classes 10 --dataset-name cifar10 --model-config ./models/configs/mobilevit.py --mode s --resize 128 --fprune-rate 0.23 --dense-model /path/to/dense/model --save-path /path/to/save/pruned/model --seed 7
```

#### Finetuning
```
python3 finetuning.py /path/to/dataset --classes 10 --dataset-name cifar10 --model-config ./models/configs/mobilevit.py --mode s --resize 128 --fprune-rate 0.23 --epoch 200 --train-ratio 0.8 --pruned-model /path/to/pruned/model --save-path /path/to/save/model --seed 7 \
```

#### Evaluate
```
python3 main.py /path/to/dataset --classes 10 --dataset-name cifar10 --model-config ./models/configs/mobilevit.py --mode s --resize 128 --epoch 200 --train-ratio 0.8 --save-path /path/to/pruned/model --fprune-rate 0.23 --test
```

## Acknowledgements
Our code is based on [ml-cvnets](https://github.com/apple/ml-cvnets).
