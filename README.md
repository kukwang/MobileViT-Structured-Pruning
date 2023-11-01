#  Compressing MobileViT using Structured Pruning

## Introduction

Apply [filter pruning](https://arxiv.org/pdf/1608.08710.pdf) to MobileViT specified in ["MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"](https://arxiv.org/abs/2110.02178).

## Experimental results

I experimented on CIFAR-10 using one GeForce RTX 3090. I applied filter pruning to the pretrained MobileViT and finetuned it for 200 epochs.

| pruning rate  | acc  | params | latency |
|:-------------:|:----:|:------:|:-------:|
| 0 %  | 89.69 % | 5.00 M | 11.44 ms |
| 20 % | 90.10 % | 4.01 M | 11.02 ms |
| 40 % | 89.73 % | 3.03 M | 11.00 ms |
| 60 % | 89.60 % | 2.06 M | 9.75 ms |
| 80 % | 88.51 % | 1.09 M | 9.37 ms |

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
This code is based on [ml-cvnets](https://github.com/apple/ml-cvnets).
