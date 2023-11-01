#  Compressing MobileViT using Structured Pruning

## Introduction

Apply [filter pruning](https://arxiv.org/pdf/1608.08710.pdf) to MobileViT specified in ["MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"](https://arxiv.org/abs/2110.02178).

## Experimental results

I experimented on CIFAR-10 using one GeForce RTX 3090. I applied filter pruning to the pretrained MobileViT-S and finetuned it for 200 epochs.

| pruning rate | acc  | latency |
|:------------:|:----:|:-------:|
| 0 %  | 89.69 % | 11.44 ms |
| 20 % | 90.10 % | 11.02 ms |
| 40 % | 89.73 % | 11.00 ms |
| 60 % | 89.60 % | 9.75 ms |
| 80 % | 88.51 % | 9.37 ms |

## Usage

#### Pruning
```
python3 filter_prune.py /path/to/dataset --classes 10 --dataset-name cifar10 \
--model-config ./models/configs/mobilevit.py --mode s --resize 128 --fprune-rate 0.23 \
--dense-model /path/to/dense/model --save-path /path/to/save/pruned/model --seed 7
```

#### Finetuning
```
python3 finetuning.py /path/to/dataset --classes 10 --dataset-name cifar10 \
--model-config ./models/configs/mobilevit.py --mode s --resize 128 --fprune-rate 0.23 \
--epoch 200 --train-ratio 0.8 \
--pruned-model /path/to/pruned/model --save-path /path/to/save/model --seed 7 \
```

#### Evaluate
```
python3 main.py /path/to/dataset --classes 10 --dataset-name cifar10 \
--model-config ./models/configs/mobilevit.py --mode s --resize 128 --fprune-rate 0.23 \
--epoch 200 --train-ratio 0.8 \
--save-path /path/to/pruned/model --test
```

## Acknowledgements
This code is based on [ml-cvnets](https://github.com/apple/ml-cvnets).
