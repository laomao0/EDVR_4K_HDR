#!/usr/bin/env bash
#
#echoho "using cuda:"$1


##CUDA_VISIBLE_DEVICES=$1
#python test_4k_hdr_with_GT_stitching.py \
#                        --input_path /DATA7_DB7/data/4khdr/data/Dataset/val_540p \
#                        --gt_path /DATA7_DB7/data/4khdr/data/Dataset/val_4k \
#                        --output_path /DATA7_DB7/data/4khdr/data/Results/001_EDVRwTSA_scratch_lr4e-4_600k_4KHDR_archived_191102-132426 \
#                        --model_path /DATA5_DB8/data/4khdr/codes/experiments/001_EDVRwTSA_scratch_lr4e-4_600k_4KHDR_archived_191102-132426/models/latest_G.pth \
#                        --gpu_id 4
#


python test_4k_hdr_with_GT_stitching.py \
        --input_path /DATA7_DB7/data/4khdr/data/Dataset/val_540p \
        --gt_path /DATA7_DB7/data/4khdr/data/Dataset/val_4k \
        --screen_notation  /DATA7_DB7/data/4khdr/data/Dataset/4khdr_frame_notation.json \
        --output_path /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_without_screen_detect \
        --model_path /DATA5_DB8/data/4khdr/codes/experiments/train_EDVR_L_Preload_30000_nobug/models/10000_G.pth \
        --opt  /DATA5_DB8/data/4khdr/codes/codes/options/train/train_EDVR_L_Preload_30000_nobug.yml \
        --gpu_id 4