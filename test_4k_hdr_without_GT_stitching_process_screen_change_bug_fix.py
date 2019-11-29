'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import time
import torch
from AverageMeter import *

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch
import itertools
import options.options as option
import argparse

import matplotlib.pyplot as plt


# def check_notation(frame_notation, folder_key, center_index):
#     """
#     check not get dis-continous frames
#     """
#     notation = frame_notation[folder_key][center_index - 2:center_index + 3]
#     if all(item == notation[0] for item in notation):
#         return 1
#     else:
#         return 0


def main():
    #################
    # configurations
    #################
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    # parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, required=True)
    parser.add_argument("--screen_notation", type=str, required=True)
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)

    PAD = 32

    total_run_time = AverageMeter()
    # print("GPU ", torch.cuda.device_count())

    device = torch.device('cuda')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print('export CUDA_VISIBLE_DEVICES=' + str(args.gpu_id))

    data_mode = 'sharp_bicubic'  # Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
    # Vid4: SR
    # REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
    #        blur (deblur-clean), blur_comp (deblur-compression).
    stage = 1  # 1 or 2, use two stage strategy for REDS dataset.
    flip_test = False

    # Input_folder = "/DATA7_DB7/data/4khdr/data/Dataset/train_sharp_bicubic"
    # GT_folder = "/DATA7_DB7/data/4khdr/data/Dataset/train_4k"
    # Result_folder = "/DATA7_DB7/data/4khdr/data/Results"

    Input_folder = args.input_path
    # GT_folder = args.gt_path
    Result_folder = args.output_path
    Model_path = args.model_path

    # create results folder
    if not os.path.exists(Result_folder):
        os.makedirs(Result_folder, exist_ok=True)



    ############################################################################
    #### model
    # if data_mode == 'Vid4':
    #     if stage == 1:
    #         model_path = '../experiments/pretrained_models/EDVR_Vimeo90K_SR_L.pth'
    #     else:
    #         raise ValueError('Vid4 does not support stage 2.')
    # elif data_mode == 'sharp_bicubic':
    #     if stage == 1:
    #         # model_path = '../experiments/pretrained_models/EDVR_REDS_SR_L.pth'
    #     else:
    #         model_path = '../experiments/pretrained_models/EDVR_REDS_SR_Stage2.pth'
    # elif data_mode == 'blur_bicubic':
    #     if stage == 1:
    #         model_path = '../experiments/pretrained_models/EDVR_REDS_SRblur_L.pth'
    #     else:
    #         model_path = '../experiments/pretrained_models/EDVR_REDS_SRblur_Stage2.pth'
    # elif data_mode == 'blur':
    #     if stage == 1:
    #         model_path = '../experiments/pretrained_models/EDVR_REDS_deblur_L.pth'
    #     else:
    #         model_path = '../experiments/pretrained_models/EDVR_REDS_deblur_Stage2.pth'
    # elif data_mode == 'blur_comp':
    #     if stage == 1:
    #         model_path = '../experiments/pretrained_models/EDVR_REDS_deblurcomp_L.pth'
    #     else:
    #         model_path = '../experiments/pretrained_models/EDVR_REDS_deblurcomp_Stage2.pth'
    # else:
    #     raise NotImplementedError

    model_path = Model_path

    if data_mode == 'Vid4':
        N_in = 7  # use N_in images to restore one HR image
    else:
        N_in = 5

    predeblur, HR_in = False, False
    back_RBs = 40
    if data_mode == 'blur_bicubic':
        predeblur = True
    if data_mode == 'blur' or data_mode == 'blur_comp':
        predeblur, HR_in = True, True
    if stage == 2:
        HR_in = True
        back_RBs = 20

    model = EDVR_arch.EDVR(nf=opt['network_G']['nf'], nframes=opt['network_G']['nframes'],
                          groups=opt['network_G']['groups'], front_RBs=opt['network_G']['front_RBs'],
                          back_RBs=opt['network_G']['back_RBs'],
                          predeblur=opt['network_G']['predeblur'], HR_in=opt['network_G']['HR_in'],
                          w_TSA=opt['network_G']['w_TSA'])

    # model = EDVR_arch.EDVR(128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)

    #### dataset
    if data_mode == 'Vid4':
        test_dataset_folder = '../datasets/Vid4/BIx4'
        GT_dataset_folder = '../datasets/Vid4/GT'
    else:
        if stage == 1:
            # test_dataset_folder = '../datasets/REDS4/{}'.format(data_mode)
            # test_dataset_folder = '/DATA/wangshen_data/REDS/val_sharp_bicubic/X4'
            test_dataset_folder = Input_folder
        else:
            test_dataset_folder = '../results/REDS-EDVR_REDS_SR_L_flipx4'
            print('You should modify the test_dataset_folder path for stage 2')
        # GT_dataset_folder = '../datasets/REDS4/GT'
        # GT_dataset_folder = '/DATA/wangshen_data/REDS/val_sharp'
        # GT_dataset_folder = GT_folder

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    if data_mode == 'Vid4' or data_mode == 'sharp_bicubic':
        padding = 'new_info'
    else:
        padding = 'replicate'
    save_imgs = True

    # save_folder = '../results/{}'.format(data_mode)
    # save_folder = '/DATA/wangshen_data/REDS/results/{}'.format(data_mode)
    save_folder = os.path.join(Result_folder, data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    # subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
    # for each subfolder
    # for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):

    end = time.time()

    # load screen change notation
    import json
    with open(args.screen_notation) as f:
        frame_notation = json.load(f)

    for subfolder in subfolder_l:

        input_subfolder = os.path.split(subfolder)[1]

        # subfolder_GT = os.path.join(GT_dataset_folder,input_subfolder)

        #if not os.path.exists(subfolder_GT):
        #    continue

        print("Evaluate Folders: ", input_subfolder)

        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)  # Num x 3 x H x W
        #img_GT_l = []
        #for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
        #    img_GT_l.append(data_util.read_img(None, img_GT_path))

        #avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):

            img_name = osp.splitext(osp.basename(img_path))[0]

            # todo here handle screen change
            select_idx, log1, log2, nota = data_util.index_generation_process_screen_change_withlog_fixbug(input_subfolder, frame_notation, img_idx, max_idx, N_in, padding=padding)

            if not log1 == None:
                logger.info('screen change')
                logger.info(nota)
                logger.info(log1)
                logger.info(log2)



            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)  # 960 x 540


            # here we split the input images 960x540 into 9 320x180 patch
            gtWidth = 3840
            gtHeight = 2160
            intWidth_ori = imgs_in.shape[4] # 960
            intHeight_ori = imgs_in.shape[3] # 540
            split_lengthY = 180
            split_lengthX = 320
            scale = 4

            intPaddingRight_ = int(float(intWidth_ori) / split_lengthX + 1) * split_lengthX - intWidth_ori
            intPaddingBottom_ = int(float(intHeight_ori) / split_lengthY + 1) * split_lengthY - intHeight_ori

            intPaddingRight_ = 0 if intPaddingRight_ == split_lengthX else intPaddingRight_
            intPaddingBottom_ = 0 if intPaddingBottom_ == split_lengthY else intPaddingBottom_

            pader0 = torch.nn.ReplicationPad2d([0, intPaddingRight_, 0, intPaddingBottom_])
            print("Init pad right/bottom " + str(intPaddingRight_) + " / " + str(intPaddingBottom_))

            intPaddingRight = PAD # 32# 64# 128# 256
            intPaddingLeft = PAD  # 32#64 #128# 256
            intPaddingTop = PAD  # 32#64 #128#256
            intPaddingBottom = PAD  # 32#64 # 128# 256



            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

            imgs_in = torch.squeeze(imgs_in, 0)# N C H W

            imgs_in = pader0(imgs_in)  # N C 540 960

            imgs_in = pader(imgs_in)  # N C 604 1024

            assert (split_lengthY == int(split_lengthY) and split_lengthX == int(split_lengthX))
            split_lengthY = int(split_lengthY)
            split_lengthX = int(split_lengthX)
            split_numY = int(float(intHeight_ori) / split_lengthY )
            split_numX = int(float(intWidth_ori) / split_lengthX)
            splitsY = range(0, split_numY)
            splitsX = range(0, split_numX)

            intWidth = split_lengthX
            intWidth_pad = intWidth + intPaddingLeft + intPaddingRight
            intHeight = split_lengthY
            intHeight_pad = intHeight + intPaddingTop + intPaddingBottom

            # print("split " + str(split_numY) + ' , ' + str(split_numX))
            y_all = np.zeros((gtHeight, gtWidth, 3), dtype="float32")  # HWC
            for split_j, split_i in itertools.product(splitsY, splitsX):
                # print(str(split_j) + ", \t " + str(split_i))
                X0 = imgs_in[:, :,
                     split_j * split_lengthY:(split_j + 1) * split_lengthY + intPaddingBottom + intPaddingTop,
                     split_i * split_lengthX:(split_i + 1) * split_lengthX + intPaddingRight + intPaddingLeft]

                # y_ = torch.FloatTensor()

                X0 = torch.unsqueeze(X0, 0)  # N C H W -> 1 N C H W

                if flip_test:
                    output = util.flipx4_forward(model, X0)
                else:
                    output = util.single_forward(model, X0)

                output_depadded = output[0, :, intPaddingTop * scale :(intPaddingTop+intHeight) * scale, intPaddingLeft * scale: (intPaddingLeft+intWidth)*scale]
                output_depadded = output_depadded.squeeze(0)
                output = util.tensor2img(output_depadded)


                y_all[split_j * split_lengthY * scale :(split_j + 1) * split_lengthY * scale,
                      split_i * split_lengthX * scale :(split_i + 1) * split_lengthX * scale, :] = \
                        np.round(output).astype(np.uint8)

                # plt.figure(0)
                # plt.title("pic")
                # plt.imshow(y_all)


            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), y_all)

            print("*****************current image process time \t " + str(
                time.time() - end) + "s ******************")
            total_run_time.update(time.time() - end, 1)

            # calculate PSNR
            #y_all = y_all / 255.
            #GT = np.copy(img_GT_l[img_idx])
            # For REDS, evaluate on RGB channels; for Vid4, evaluate on the Y channel
            #if data_mode == 'Vid4':  # bgr2y, [0, 1]
            #    GT = data_util.bgr2ycbcr(GT, only_y=True)
            #    y_all = data_util.bgr2ycbcr(y_all, only_y=True)

            #y_all, GT = util.crop_border([y_all, GT], crop_border)
            #crt_psnr = util.calculate_psnr(y_all * 255, GT * 255)
            #logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB'.format(img_idx + 1, img_name, crt_psnr))

            logger.info('{} : {:3d} - {:25} \t'.format(input_subfolder, img_idx + 1, img_name))

            #if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
            #    avg_psnr_center += crt_psnr
            #    N_center += 1
            #else:  # border frames
            #    avg_psnr_border += crt_psnr
            #    N_border += 1

        #avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        #avg_psnr_center = avg_psnr_center / N_center
        #avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        #avg_psnr_l.append(avg_psnr)
        #avg_psnr_center_l.append(avg_psnr_center)
        #avg_psnr_border_l.append(avg_psnr_border)

        #logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
        #            'Center PSNR: {:.6f} dB for {} frames; '
        #            'Border PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_psnr,
        #                                                           (N_center + N_border),
        #                                                           avg_psnr_center, N_center,
        #                                                           avg_psnr_border, N_border))

    #logger.info('################ Tidy Outputs ################')
    #for subfolder_name, psnr, psnr_center, psnr_border in zip(subfolder_name_l, avg_psnr_l,
    #                                                          avg_psnr_center_l, avg_psnr_border_l):
    #    logger.info('Folder {} - Average PSNR: {:.6f} dB. '
    #                'Center PSNR: {:.6f} dB. '
    #                'Border PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center,
    #                                                 psnr_border))
    #logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    #logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
    #            'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
    #                sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
    #                sum(avg_psnr_center_l) / len(avg_psnr_center_l),
    #                sum(avg_psnr_border_l) / len(avg_psnr_border_l)))


if __name__ == '__main__':
    main()
