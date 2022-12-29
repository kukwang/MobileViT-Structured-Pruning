# dataset path
# cifar10: /home/user/dataset/cifar10
# cifar100: /home/user/dataset/cifar100
# ImageNet: /home/user/dataset/ILSVRC2012

dataset_name=cifar10
path=/home/user/dataset/cifar10
config=./models/configs/mobilevit.py

# 1 gpu cifar100
CUDA_VISIBLE_DEVICES=0 python3 main.py $path --classes 10 --dataset-name $dataset_name --model-config $config \
--mode xxs --resize 128 --epoch 50 --train-ratio 0.8 \
| tee log_128_val_split.txt

# # 2 gpu
# CUDA_VISIBLE_DEVICES=0,1 python3 main.py $path --classes 100 --model-config $config --mode xxs --epoch 30