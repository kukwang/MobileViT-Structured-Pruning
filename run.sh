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

seed=7
dense_path=/home/user/kwangsoo/paper_codes/my_paper/KICS-2023-mine/save/mobilevit_${mode}_${dataset_name}_${ep}ep_dense_cosdecay_real.pth


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

seed=7
dense_path=/home/user/kwangsoo/paper_codes/my_paper/KICS-2023-mine/save/mobilevit_${mode}_${dataset_name}_${ep}ep_dense_cosdecay_real.pth
# 1 gpu mobilevit cifar10 testing
CUDA_VISIBLE_DEVICES=2 python3 training.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
--model-config ${config} --mode ${mode} --resize ${resize} --epoch ${ep} --train-ratio ${train_ratio} \
--save-path ${dense_path} --test \
| tee log_${mode}_${resize}_${ep}ep_dense_seed${seed}_cosdecay_real_test_latency.txt

# real sp: 0.2 0.4 0.6 0.8
for pr in 0.11 0.23 0.37 0.55
do
    pruned_path=/home/user/kwangsoo/paper_codes/my_paper/KICS-2023-mine/save/mobilevit_${mode}_${dataset_name}_${ep}ep_pr${pr}_seed${seed}_cosdecay_real.pth
    finetuning_path=/home/user/kwangsoo/paper_codes/my_paper/KICS-2023-mine/save/mobilevit_${mode}_${dataset_name}_${ep}ep_pr${pr}_seed${seed}_finetuning_cosdecay_real.pth

    # 1 gpu mobilevit cifar10 testing
    CUDA_VISIBLE_DEVICES=2 python3 training.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
    --model-config ${config} --mode ${mode} --resize ${resize} --epoch ${ep} --train-ratio ${train_ratio} \
    --save-path ${finetuning_path} --fprune-rate ${pr} --test \
    | tee log_${mode}_${resize}_${ep}ep_pr${pr}_seed${seed}_cosdecay_real_test_latency.txt

    # mv log_${mode}_${resize}_${ep}ep_pr${pr}_cosdecay_real_latency.txt ./logs

    # # 1 gpu mobilevit cifar10 pruning without testing
    # CUDA_VISIBLE_DEVICES=0 python3 filter_prune.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
    # --model-config ${config} --mode ${mode} --resize ${resize} --fprune-rate ${pr} --dense-model ${dense_path} \
    # --save-path ${pruned_path} --seed ${seed} \
    # | tee log_${mode}_${resize}_${ep}ep_pr${pr}_seed${seed}_cosdecay_real.txt

    # mv log_${mode}_${resize}_${ep}ep_pr${pr}_seed${seed}_cosdecay_real.txt ./logs

    # # 1 gpu mobilevit cifar10 finetuning without knowledge distillation
    # CUDA_VISIBLE_DEVICES=0 python3 finetuning.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
    # --model-config ${config} --mode ${mode} --resize ${resize} --fprune-rate ${pr} --epoch ${fep} --train-ratio ${train_ratio} \
    # --dense-model ${dense_path} --pruned-model ${pruned_path} --save-path ${finetuning_path} --seed ${seed} \
    # | tee log_${mode}_${resize}_${fep}_pr${pr}_seed${seed}_finetuning_cosdecay_real.txt

    # mv log_${mode}_${resize}_${fep}_pr${pr}_seed${seed}_finetuning_cosdecay_real.txt ./logs

done
# ========================================================================================================================

# pr=0.37
# seed=20
# pruned_path=/home/user/kwangsoo/paper_codes/my_paper/KICS-2023-mine/save/mobilevit_${mode}_${dataset_name}_${ep}ep_pr${pr}.pth

# # 1 gpu mobilevit cifar10 finetuning without knowledge distillation
# CUDA_VISIBLE_DEVICES=0 python3 finetuning.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
# --model-config ${config} --mode ${mode} --resize ${resize} --fprune-rate ${pr} --epoch ${fep} --train-ratio ${train_ratio} \
# --dense-model ${dense_path} --pruned-model ${pruned_path} --seed ${seed} \
# | tee log_${mode}_${resize}_${fep}_pr${pr}_finetuning_cosdecay.txt

# mv log_${mode}_${resize}_${fep}_pr${pr}_finetuning_cosdecay.txt ./logs

# # 2 gpu
# CUDA_VISIBLE_DEVICES=0,1 python3 main.py $path --classes 100 --model-config $config --mode xxs --epoch 30


# # 1 gpu mobilevit cifar10 pruning + test
# CUDA_VISIBLE_DEVICES=0 python3 filter_prune.py ${dataset_path} --classes ${classes} --dataset-name ${dataset_name} \
# --model-config ${config} --mode ${mode} --resize ${resize} --fprune-rate ${pr} --dense-model ${dense_path} \
# --save-path ${pruned_path} \
# --test --train-ratio ${train_ratio} --test-batch-size 128 \
# | tee log_${resize}_${ep}ep_pr${pr}.txt

# mv log_${resize}_${ep}ep_pr${pr}.txt ./logs
