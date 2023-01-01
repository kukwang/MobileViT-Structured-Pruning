# dataset path
# cifar10: /home/user/dataset/cifar10
# cifar100: /home/user/dataset/cifar100
# ImageNet: /home/user/dataset/ILSVRC2012

dataset_name=cifar10
dataset_path=/home/user/dataset/cifar10
config=./models/configs/mobilevit.py
classes=10
mode=s
resize=128
ep=200
train_ratio=0.8

pr=0.5
dense_path=/home/user/kwangsoo/paper_codes/my_paper/KICS-2023-mine/save/mobilevit_${mode}_${dataset_name}_${ep}ep_dense.pth

# # 1 gpu cifar100 test-only
# CUDA_VISIBLE_DEVICES=0 python3 main.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
# --model-config ${config} --mode ${mode} --resize ${resize} --epoch ${ep} --train-ratio ${train_ratio} --test --test-batch-size 128 \
# | tee log_${resize}_${ep}ep.txt

# 1 gpu cifar100
CUDA_VISIBLE_DEVICES=0 python3 main.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
--model-config ${config} --mode ${mode} --resize ${resize} --epoch ${ep} --train-ratio ${train_ratio} \
| tee log_${mode}_${resize}_${ep}ep_dense.txt

mv log_${mode}_${resize}_${ep}ep_dense.txt ./logs

# mode=xs
# CUDA_VISIBLE_DEVICES=0 python3 main.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
# --model-config ${config} --mode ${mode} --resize ${resize} --epoch ${ep} --train-ratio ${train_ratio} \
# | tee log_${mode}_${resize}_${ep}ep_dense.txt

# mv log_${mode}_${resize}_${ep}ep_dense.txt ./logs

# mode=xxs
# CUDA_VISIBLE_DEVICES=0 python3 main.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
# --model-config ${config} --mode ${mode} --resize ${resize} --epoch ${ep} --train-ratio ${train_ratio} \
# | tee log_${mode}_${resize}_${ep}ep_dense.txt

# mv log_${mode}_${resize}_${ep}ep_dense.txt ./logs

# CUDA_VISIBLE_DEVICES=0 python3 filter_prune.py --classes ${classes} --dataset-name ${dataset_name} \
# --model-config ${config} --mode ${mode} --resize ${resize} --fprune-rate ${pr} --dense-model ${dense_path} \
# | tee log_${resize}_${ep}ep_prune.txt

# mv log_${resize}_${ep}ep_prune.txt ./logs

# # 2 gpu
# CUDA_VISIBLE_DEVICES=0,1 python3 main.py $path --classes 100 --model-config $config --mode xxs --epoch 30