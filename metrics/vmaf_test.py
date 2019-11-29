import os
import json
# import math
# import argparse
# import random
# import logging
#
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from data.data_sampler import DistIterSampler
#
# import options.options as option
# from utils import util
# import data.util as data_util
# from data import create_dataloader, create_dataset
# from models import create_model
# from AverageMeter import *

val_name = ['23381522', '62600438', '63056025']

def val_vmaf(input_path_, gt_path_, mp4_save_path, model_path):
    for i, item_name in enumerate(val_name):

        mp4_file = item_name + '.mp4'
        mp4_save_path_ = os.path.join(mp4_save_path, mp4_file)

        if os.path.exists(mp4_save_path_):
            os.remove(mp4_save_path_)

        if i == 0:
            retn = os.system(
                '/mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/ffmpeg/ffmpeg -r 24000/1001 -i {}/{}_%02d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 {}'.format(
                    input_path_,
                    item_name,
                    mp4_save_path_))
        else:
            retn = os.system(
                '/mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/ffmpeg/ffmpeg -r 24000/1001 -i {}/{}_{}%02d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 {}'.format(
                    input_path_,
                    item_name,
                    str(i),
                    mp4_save_path_))

        print (item_name, ' has been transformed to mp4')


        reference_path = os.path.join(gt_path_, mp4_file)
        distorted_path = os.path.join(mp4_save_path, mp4_file)

        vmaf_value_f = os.popen(
            '/mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/vmaf/ffmpeg2vmaf 1840 2160 {} {} --model {} --out-fmt json'.format(
                reference_path,
                distorted_path,
                model_path
            )
        )

        vmaf_value = vmaf_value_f.read()
        vmaf_value = json.loads(vmaf_value)
        vmaf_value_f.close()

        #print (vmaf_value['aggregate']['VMAF_score'])
        print (item_name, ' vmaf value: ', vmaf_value['aggregate']['VMAF_score'])


    return 0






if __name__ == "__main__":

    gt_path = '/mnt/lustre/shanghai/cmic/home/xyz18/Dataset/val_4k/'
    gt_vmaf_path = '/mnt/lustre/shanghai/cmic/home/xyz18/raw/SDR_4k/'
    input_path = '/mnt/lustre/shanghai/cmic/home/xyz18/experiments/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_val_test_11/val_images/'
    mp4_save_path = '/mnt/lustre/shanghai/cmic/home/xyz18/experiments/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_val_test_11/val_mp4/'
    model_path = '/mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/vmaf/model/vmaf_4k_v0.6.1.pkl'

    if not os.path.exists(mp4_save_path):
        os.mkdir(mp4_save_path)

    val_vmaf(input_path, gt_vmaf_path, mp4_save_path, model_path)







