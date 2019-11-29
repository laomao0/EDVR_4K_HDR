#!/usr/bin/env bash

device=$1 #the first arg is the device num.
echo Using CUDA device $device

#CUDA_VISIBLE_DEVICES=$device python train.py \
#        --opt /DATA5_DB8/data/4khdr/codes/codes/options/train/my_train_EDVR_L.yml

CUDA_VISIBLE_DEVICES=$device python train.py \
        --opt /DATA5_DB8/data/4khdr/codes/codes/options/train/my_train_EDVR_L_debug.yml