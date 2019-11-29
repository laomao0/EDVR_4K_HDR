#!/usr/bin/env bash

#device=$1 #the first arg is the device num.
#echo Using CUDA device $device

#CUDA_VISIBLE_DEVICES=$device python train.py \
#        --opt /DATA5_DB8/data/4khdr/codes/codes/options/train/my_train_EDVR_L.yml



#python -m torch.distributed.launch \
#        --nproc_per_node=2 \
#        --master_port=1234 \
#        train_parallel.py \
#        --opt /DATA7_DB7/data/4khdr/codes/codes/codes/options/train/train_EDVR_L_Preload_30000_nobug_4_parral.yml \
#        --launcher pytorch

#CUDA_VISIBLE_DEVICES=2,3,4,5
#      --master_port=1245 \
#CUDA_VISIBLE_DEVICES=5,6
python -m torch.distributed.launch \
      --nproc_per_node=4 \
      --master_port=1589 \
      train_parallel.py \
      --opt /DATA7_DB7/data/4khdr/codes/codes/codes/options/train/train_EDVR_L_Preload_30000_nobug_ssim_4_LR_1e-5_parral.yml \
      --launcher pytorch