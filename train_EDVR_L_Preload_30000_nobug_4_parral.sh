#!/usr/bin/env bash

#device=$1 #the first arg is the device num.
#echo Using CUDA device $device

#CUDA_VISIBLE_DEVICES=$device python train.py \
#        --opt /DATA5_DB8/data/4khdr/codes/codes/options/train/my_train_EDVR_L.yml

python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=1234 \
        train_parallel.py \
        --launcher pytorch \
        --opt /DATA7_DB7/data/4khdr/codes/codes/codes/options/train/train_EDVR_L_Preload_30000_nobug_4_parral.yml \
