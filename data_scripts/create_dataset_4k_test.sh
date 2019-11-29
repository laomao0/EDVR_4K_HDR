#!/usr/bin/env bash
#
#echoho "using cuda:"$1


#CUDA_VISIBLE_DEVICES=$1
python create_dataset_4k.py \
                        --ffmpeg_dir /DATA5_DB8/data/4khdr/codes/codes/ffmpeg \
                        --dataset hdr\
                        --window_size 1 \
                        --enable_4k 0 \
                        --enable_540p 0 \
                        --enable_540p_test 1 \
                        --dataset_folder  /DATA7_DB7/data/4khdr/data/Dataset \
                        --train_4k_video_path  /DATA7_DB7/data/4khdr/data/SDR_4k \
                        --train_540p_video_path  /DATA7_DB7/data/4khdr/data/SDR_540p \
                        --test_540p_video_path /DATA7_DB7/data/4khdr/data/TEST_SDR_540p \
                        --img_width_gt 3840 \
                        --img_height_gt 2160 \
                        --img_width_input 960 \
                        --img_height_input 540


