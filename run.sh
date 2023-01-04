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
fep=200
train_ratio=0.8
kd_lambda=0.1

dense_path=/home/user/kwangsoo/paper_codes/my_paper/KICS-2023-mine/save/mobilevit_${mode}_${dataset_name}_${ep}ep_dense.pth

pr=0.29
pruned_path=/home/user/kwangsoo/paper_codes/my_paper/KICS-2023-mine/save/mobilevit_${mode}_${dataset_name}_${ep}ep_pr${pr}.pth

# # 1 gpu mobilevit cifar10 training
# CUDA_VISIBLE_DEVICES=0 python3 training.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
# --model-config ${config} --mode ${mode} --resize ${resize} --epoch ${ep} --train-ratio ${train_ratio} \
# | tee log_${mode}_${resize}_${ep}ep_dense.txt

# mv log_${mode}_${resize}_${ep}ep_dense.txt ./logs

# # 1 gpu mobilevit cifar10 pruning without testing
# CUDA_VISIBLE_DEVICES=0 python3 filter_prune.py --classes ${classes} --dataset-name ${dataset_name} \
# --model-config ${config} --mode ${mode} --resize ${resize} --fprune-rate ${pr} --dense-model ${dense_path} \
# --save-path ${pruned_path} \
# | tee log_${resize}_${ep}ep_pr${pr}.txt

# mv log_${resize}_${ep}ep_pr${pr}.txt ./logs

# # 1 gpu mobilevit cifar10 finetuning without knowledge distillation
# CUDA_VISIBLE_DEVICES=0 python3 finetuning.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
# --model-config ${config} --mode ${mode} --resize ${resize} --fprune-rate ${pr} --epoch ${fep} --train-ratio ${train_ratio} \
# --dense-model ${dense_path} --pruned-model ${pruned_path} \
# | tee log_${mode}_${resize}_${fep}_pr${pr}_finetuning.txt

# mv log_${mode}_${resize}_${fep}_pr${pr}_finetuning.txt ./logs

# # 1 gpu mobilevit cifar10 finetuning with knowledge distillation
# CUDA_VISIBLE_DEVICES=0 python3 finetuning.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
# --model-config ${config} --mode ${mode} --resize ${resize} --fprune-rate ${pr} --epoch ${fep} --train-ratio ${train_ratio} \
# --dense-model ${dense_path} --pruned-model ${pruned_path} \
# --kd-lambda ${kd_lambda} \
# | tee log_${mode}_${resize}_${fep}_pr${pr}_finetuning_kd.txt

# mv log_${mode}_${resize}_${fep}_pr${pr}_finetuning_kd.txt ./logs

# ========================================================================================================================

# loop
# real sp: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
for pr in 0.05 0.11 0.16 0.23 0.29 0.37 0.45 0.55 0.68
do
    pruned_path=/home/user/kwangsoo/paper_codes/my_paper/KICS-2023-mine/save/mobilevit_${mode}_${dataset_name}_${ep}ep_pr${pr}.pth

    # # 1 gpu mobilevit cifar10 pruning with testing
    # CUDA_VISIBLE_DEVICES=0 python3 filter_prune.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
    # --model-config ${config} --mode ${mode} --resize ${resize} --fprune-rate ${pr} --dense-model ${dense_path} \
    # --save-path ${pruned_path} \
    # --test --train-ratio ${train_ratio} --test-batch-size 1 \
    # | tee log_${resize}_${ep}ep_pr${pr}.txt

    # mv log_${resize}_${ep}ep_pr${pr}.txt ./logs

#     # 1 gpu mobilevit cifar10 finetuning without knowledge distillation
#     CUDA_VISIBLE_DEVICES=0 python3 finetuning.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
#     --model-config ${config} --mode ${mode} --resize ${resize} --fprune-rate ${pr} --epoch ${fep} --train-ratio ${train_ratio} \
#     --dense-model ${dense_path} --pruned-model ${pruned_path} \
#     | tee log_${mode}_${resize}_${fep}_pr${pr}_finetuning.txt

#     mv log_${mode}_${resize}_${fep}_pr${pr}_finetuning.txt ./logs

    # 1 gpu mobilevit cifar10 finetuning with knowledge distillation
    CUDA_VISIBLE_DEVICES=0 python3 finetuning.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
    --model-config ${config} --mode ${mode} --resize ${resize} --fprune-rate ${pr} --epoch ${fep} --train-ratio ${train_ratio} \
    --dense-model ${dense_path} --pruned-model ${pruned_path} \
    --kd-lambda ${kd_lambda} \
    | tee log_${mode}_${resize}_${fep}_pr${pr}_finetuning_kd.txt

    mv log_${mode}_${resize}_${fep}_pr${pr}_finetuning_kd.txt ./logs

done
# ========================================================================================================================

# # 2 gpu
# CUDA_VISIBLE_DEVICES=0,1 python3 main.py $path --classes 100 --model-config $config --mode xxs --epoch 30

# # 1 gpu cifar100 test-only
# CUDA_VISIBLE_DEVICES=0 python3 training.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
# --model-config ${config} --mode ${mode} --resize ${resize} --epoch ${ep} --train-ratio ${train_ratio} --test --test-batch-size 128 \
# | tee log_${resize}_${ep}ep.txt

# # 1 gpu mobilevit cifar10 pruning + test
# CUDA_VISIBLE_DEVICES=0 python3 filter_prune.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
# --model-config ${config} --mode ${mode} --resize ${resize} --fprune-rate ${pr} --dense-model ${dense_path} \
# --save-path ${pruned_path} \
# --test --train-ratio ${train_ratio} --test-batch-size 128 \
# | tee log_${resize}_${ep}ep_pr${pr}.txt

# mv log_${resize}_${ep}ep_pr${pr}.txt ./logs
