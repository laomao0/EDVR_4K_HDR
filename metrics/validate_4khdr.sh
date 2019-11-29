#!/usr/bin/env bash

#python validate_4khdr.py \
##    --gt_path  /mnt/lustre/shanghai/cmic/home/xyz18/Dataset/val_4k/ \
##    --input_path  /mnt/lustre/shanghai/cmic/home/xyz18/experiments/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_val_test_11/val_images/ \
##    --gt_vmaf_path  /mnt/lustre/shanghai/cmic/home/xyz18/raw/SDR_4k/ \
##    --mp4_save_path  /mnt/lustre/shanghai/cmic/home/xyz18/experiments/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_val_test_11/val_mp4/ \
##    --model_path  /mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/vmaf/model/vmaf_4k_v0.6.1.pkl \
##    --log_path  /mnt/lustre/shanghai/cmic/home/xyz18/experiments/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_val_test_11/

python validate_4khdr.py \
    --gt_path  /mnt/lustre/shanghai/cmic/home/xyz18/Dataset/val_4k/ \
    --input_path  /mnt/lustre/shanghai/cmic/home/xyz18/experiments/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_5/val_images/ \
    --gt_vmaf_path  /mnt/lustre/shanghai/cmic/home/xyz18/raw/SDR_4k/ \
    --mp4_save_path  /mnt/lustre/shanghai/cmic/home/xyz18/experiments/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_5/val_mp4/ \
    --model_path  /mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/vmaf/model/vmaf_4k_v0.6.1.pkl \
    --log_path  /mnt/lustre/shanghai/cmic/home/xyz18/experiments/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_5/