#!/usr/bin/env bash
#
#device=$1 #the first arg is the device num.
#echo Using CUDA device $device

#python test_4k_hdr_with_GT_stitching_process_screen_change.py \
#        --input_path /DATA7_DB7/data/4khdr/data/Dataset/val_540p \
#        --gt_path /DATA7_DB7/data/4khdr/data/Dataset/val_4k \
#        --screen_notation  /DATA7_DB7/data/4khdr/data/Dataset/4khdr_frame_notation.json \
#        --output_path /DATA7_DB7/data/4khdr/data/Results/001_EDVRwTSA_scratch_lr4e-4_600k_4KHDR_archived_191102-132426-screen-detect \
#        --model_path /DATA5_DB8/data/4khdr/codes/experiments/001_EDVRwTSA_scratch_lr4e-4_600k_4KHDR_archived_191102-132426/models/latest_G.pth \
#        --opt  /DATA5_DB8/data/4khdr/codes/codes/options/train/my_train_EDVR_L.yml \
#        --gpu_id 4
#


#CUDA_VISIBLE_DEVICES=$1
#CUDA_VISIBLE_DEVICES=$device

#python test_4k_hdr_with_GT_stitching_process_screen_change.py \
#        --input_path /DATA7_DB7/data/4khdr/data/Dataset/val_540p \
#        --gt_path /DATA7_DB7/data/4khdr/data/Dataset/val_4k \
#        --screen_notation  /DATA7_DB7/data/4khdr/data/Dataset/4khdr_frame_notation.json \
#        --output_path /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug \
#        --model_path /DATA5_DB8/data/4khdr/codes/experiments/train_EDVR_L_Preload_30000_nobug/models/10000_G.pth \
#        --opt  /DATA5_DB8/data/4khdr/codes/codes/options/train/train_EDVR_L_Preload_30000_nobug.yml \
#        --gpu_id 4

python test_4k_hdr_with_GT_stitching_process_screen_change.py \
        --input_path /DATA7_DB7/data/4khdr/data/Dataset/val_540p \
        --gt_path /DATA7_DB7/data/4khdr/data/Dataset/val_4k \
        --screen_notation  /DATA7_DB7/data/4khdr/data/Dataset/4khdr_frame_notation.json \
        --output_path /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_20000_G \
        --model_path /DATA5_DB8/data/4khdr/codes/experiments/train_EDVR_L_Preload_30000_nobug/models/20000_G.pth \
        --opt  /DATA5_DB8/data/4khdr/codes/codes/options/train/train_EDVR_L_Preload_30000_nobug.yml \
        --gpu_id 5


