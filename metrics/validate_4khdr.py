import os
import json
import argparse
import sys
sys.path.append('/mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/')
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
from utils import util
import data.util as data_util
# from data import create_dataloader, create_dataset
# from models import create_model
# from AverageMeter import *

#val_name = ['23381522', '62600438', '63056025']

def val_vmaf(input_path_, gt_path_, mp4_save_path, model_path, val_name):
    val_vmaf_dic = {}
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
            '/mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/vmaf/ffmpeg2vmaf 3840 2160 {} {} --model {} --out-fmt json'.format(
                reference_path,
                distorted_path,
                model_path
            )
        )

        vmaf_value = vmaf_value_f.read()
        vmaf_value = json.loads(vmaf_value)
        vmaf_value_f.close()

        val_vmaf_dic[item_name] = vmaf_value['aggregate']['VMAF_score']

        #print (vmaf_value['aggregate']['VMAF_score'])
        print (item_name, ' vmaf value: ', vmaf_value['aggregate']['VMAF_score'])


    return val_vmaf_dic



def val_4khdr(input_path_, gt_path_, gt_vmaf_path, mp4_save_path, model_path, log_path_):

    #log_path = os.path.join(input_path_, '..',"eval_log.txt")
    log_path = os.path.join(log_path_, "eval_log.txt")
    if os.path.exists(log_path):
        os.remove(log_path)
        os.mknod(log_path)
    else:
        os.mknod(log_path)

    dict_all = {}
    val_name = []
    #imgs = sorted(os.listdir(input_path_))[0:10] # todo for test
    imgs = sorted(os.listdir(input_path_))
    pbar = util.ProgressBar(len(imgs))
    for idx_d, img in enumerate(imgs):

        folder, name = img.split('_') # folder, and image name
        pbar.update('Test {} - {}/{}'.format(folder, idx_d, len(imgs)))
        if not folder in dict_all:
            dict_all[folder] = {}
            dict_all[folder]['psnr'] = []
            dict_all[folder]['ssim'] = []
            dict_all[folder]['vmaf'] = []
            val_name.append(folder)


        img_path = os.path.join(input_path_, img)
        #print(name)
        gt_path = os.path.join(gt_path_, folder, '{:05d}.png'.format(int(name[-6:-4])))
        #print (gt_path)
        img_in = data_util.read_img(env=None, path=img_path, norm=False)
        img_gt = data_util.read_img(env=None, path=gt_path, norm=False)

        psnr = util.calculate_psnr(img_in, img_gt)
        ssim = util.calculate_ssim(img_in, img_gt)

        dict_all[folder]['psnr'].append(psnr)
        dict_all[folder]['ssim'].append(ssim)

        pstring = "Folder {:s} Img {:s} PSNR {:.4f} dB, SSIM {:.4f}".format(folder, name, psnr, ssim)
        print(pstring)
        print(pstring, file=open(os.path.join(log_path), "a"))



    pstring = "------------------In summary -------------------------------"
    print(pstring)
    print(pstring, file=open(os.path.join(log_path), "a"))




    # todo: add VMAF
    #VMAF = {}
    VMAF = val_vmaf(input_path_, gt_vmaf_path, mp4_save_path, model_path, val_name)
    #for folder in dict_all:

        # todo compute VMAF
        #VMAF_value = xxx

        #VMAF[folder] = VMAF_value

    # todo: end of VMAF



    final_score = 0

    for folder in dict_all:
        psnr_folder = sum(dict_all[folder]['psnr'])/len(dict_all[folder]['psnr'])
        ssim_folder = sum(dict_all[folder]['ssim'])/len(dict_all[folder]['ssim'])
        dict_all[folder]['psnr'].append(psnr_folder)
        dict_all[folder]['ssim'].append(ssim_folder)
        pstring = "Average PSNR for {:s} is {:.4f} dB".format(folder, psnr_folder)
        print(pstring)
        print(pstring, file=open(os.path.join(log_path), "a"))
        pstring = "Average SSIM for {:s} is {:.4f} ".format(folder, ssim_folder)
        print(pstring)
        print(pstring, file=open(os.path.join(log_path), "a"))
        VMAF_value =  VMAF[folder]
        pstring = "Average VMAF for {:s} is {:.4f} ".format(folder, VMAF_value)
        print(pstring)
        print(pstring, file=open(os.path.join(log_path), "a"))
        socre = psnr_folder * 25 / 50 + (ssim_folder - 0.4) * 25 / 0.6 + VMAF_value * 50 / 80
        pstring = "Score for {:s} is {:.4f} ".format(folder, socre)
        print(pstring)
        print(pstring, file=open(os.path.join(log_path), "a"))
        print ('\n')
        print('\n', file=open(os.path.join(log_path), "a"))
        final_score += socre

    final_score = final_score / len(val_name)

    pstring = 10 * '-' + 'Final Score' + 10 * '-'
    print(pstring)
    print(pstring, file=open(os.path.join(log_path), "a"))
    pstring = "Final score is {:.4f} ".format(final_score)
    print(pstring)
    print(pstring, file=open(os.path.join(log_path), "a"))



    # todo




# test code

if __name__ == "__main__":

    # gt_path = '/mnt/lustre/shanghai/cmic/home/xyz18/Dataset/val_4k/'
    # input_path = '/mnt/lustre/shanghai/cmic/home/xyz18/experiments/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_val_test_11/val_images/'
    # gt_vmaf_path = '/mnt/lustre/shanghai/cmic/home/xyz18/raw/SDR_4k/'
    # mp4_save_path = '/mnt/lustre/shanghai/cmic/home/xyz18/experiments/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_val_test_11/val_mp4/'
    # model_path = '/mnt/lustre/shanghai/cmic/home/xyz18/codes/codes/codes/vmaf/model/vmaf_4k_v0.6.1.pkl'
    # log_path = '/mnt/lustre/shanghai/cmic/home/xyz18/experiments/train_EDVR_L_Preload_30000_nobug_ssim_predeblur_parral_suzhou_val_test_11/'

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, required=True, help='path to gt')
    parser.add_argument("--input_path", type=str, required=True, help='path to the lq')
    parser.add_argument("--gt_vmaf_path", type=str, required=True, help='path to the gt vmaf')
    parser.add_argument("--mp4_save_path", type=str, required=True, help='path to mp4_save')
    parser.add_argument("--model_path", type=str, required=True, help='path to vmaf model')
    parser.add_argument("--log_path", type=str, required=True, help='path to log')
    args = parser.parse_args()

    gt_path = args.gt_path
    input_path = args.input_path
    gt_vmaf_path = args.gt_vmaf_path
    mp4_save_path = args.mp4_save_path
    model_path = args.model_path
    log_path = args.log_path

    if not os.path.exists(mp4_save_path):
        os.mkdir(mp4_save_path)

    val_4khdr(input_path, gt_path, gt_vmaf_path, mp4_save_path, model_path, log_path)







