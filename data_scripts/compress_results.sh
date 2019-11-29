#!/usr/bin/env bash

#python compress_results.py \
#    --ffmpeg_dir /DATA5_DB8/data/4khdr/codes/codes/ffmpeg \
#    --imgs_folder_input /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug/sharp_bicubic\
#    --output_folder /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug/\
#    --bitrate 150\

#python compress_results.py \
#    --ffmpeg_dir /DATA5_DB8/data/4khdr/codes/codes/ffmpeg \
#    --imgs_folder_input /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug/sharp_bicubic \
#    --output_folder /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug/Compressed_r25_v120 \
#    --bitrate 120 \

# python compress_results.py \
#     --ffmpeg_dir /DATA5_DB8/data/4khdr/codes/codes/ffmpeg \
#     --imgs_folder_input /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/sharp_bicubic \
#     --output_folder /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/compressed_120 \
#     --bitrate 120 \

# 1106, yjpan
# python compress_results.py \
#     --ffmpeg_dir /DATA5_DB8/data/4khdr/codes/codes/ffmpeg \
#     --imgs_folder_input /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/sharp_bicubic \
#     --output_folder     /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/compressed_120 \
#     --bitrate 120 \

# /DATA5_DB8/data/4khdr/codes/codes/ffmpeg/ffmpeg -r 25 -i  /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/sharp_bicubic/54003182/%05d.png   -b:v 100M -vcodec libx265 /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/compressed_120/54003182.mp4

## 56459353
#/DATA5_DB8/data/4khdr/codes/codes/ffmpeg/ffmpeg -r 25 -i  /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/sharp_bicubic/$1/%05d.png   -b:v $2M -vcodec libx265 /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/compressed_120/$1.mp4


#python compress_results.py \
#    --ffmpeg_dir ffmpeg \
#    --imgs_folder_input /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/sharp_bicubic \
#    --output_folder /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_221_20000_G_test_540p/compressed_yuv422p \
#    --bitrate 10 \



#python compress_results.py \
#    --ffmpeg_dir ffmpeg \
#    --imgs_folder_input /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_ssim_4_lr_1e-5_latest_G_test_540p_fixbug_new/sharp_bicubic \
#    --output_folder /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_ssim_4_lr_1e-5_latest_G_test_540p_fixbug_new/compressed_yuv422p \
#    --bitrate 9 \

python compress_results.py \
    --ffmpeg_dir ffmpeg \
    --imgs_folder_input /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_nf192_rb80_ssim_parall/sharp_bicubic \
    --output_folder /DATA7_DB7/data/4khdr/data/Results/train_EDVR_L_Preload_30000_nobug_nf192_rb80_ssim_parall/compressed_yuv422p \
    --bitrate 9 \
