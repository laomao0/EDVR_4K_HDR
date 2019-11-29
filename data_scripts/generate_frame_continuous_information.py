####
# check continuous
# for example   f0 f1 f2 f3 f4 f5   6 frames, f0 f1 f2 continuous, f3 f4 f5 continous,
#  the script:   1  1  1  0  0 0
####

import cv2
import time
import sys
import os
import random
import numpy
import scipy
import scipy.stats
from scipy.misc import imread, imsave, imshow, imresize, imsave
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import argparse

# Make the output through Images.....
# OutputDir = "/DATA7_DB7/data/4khdr/data/Dataset/"
# # Input540pDir = "/DATA7_DB7/data/4khdr/data/Dataset/train_540p"
# Input540pDir = "/DATA7_DB7/data/4khdr/data/Dataset/test_540p"
# save_name = "4khdr_frame_notation"

parser = argparse.ArgumentParser()
parser.add_argument("--OutputDir", type=str, required=True)
parser.add_argument("--Input540pDir", type=str, required=True)
parser.add_argument("--save_name", type=str, required=True)
args = parser.parse_args()


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


##-------------------Start Processing-----------------------------
def process(Input540pDir, OutputDir, save_name):
    Noation = {}

    folder_lists = sorted(os.listdir(Input540pDir))

    Read = 0

    if Read == 0:

        for folder_index, folder_ in enumerate(folder_lists):

            print("processing: ", folder_)

            Noation[folder_] = []

            folder_path = os.path.join(Input540pDir, folder_)
            file_list = sorted(os.listdir(folder_path))

            Num = 10
            Len = len(file_list)
            Len_div_Num = int(Len / Num)

            for i in range(Len_div_Num):

                file_list_div = file_list[i * Num:(i + 1) * Num]

                PSNR_total = 0

                for indx, file in enumerate(file_list_div):
                    # i = 0
                    # print(file)
                    # if os.path.isdir(file):
                    #     continue
                    #
                    if indx >= len(file_list_div) - 1:
                        break

                    current_file_path = os.path.join(folder_path, file)
                    next_name = '{:05d}.png'.format((int(file[:-4]) + 1))
                    next_file_path = os.path.join(folder_path, next_name)

                    current_img = cv2.imread(current_file_path, cv2.IMREAD_UNCHANGED)  # HWC
                    next_img = cv2.imread(next_file_path, cv2.IMREAD_UNCHANGED)

                    current_img_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
                    next_img_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

                    # calculate the optical flow to determine if it is too normal sample

                    # current_img_gray = cv2.resize(current_img, tuple(resize_input))
                    # next_img_gray = cv2.resize(next_img_gray, tuple(resize_input))

                    psnr = calculate_psnr(current_img_gray, next_img_gray)

                    # print(file, next_name, 'psnr', psnr)

                    PSNR_total = PSNR_total + psnr

                Avg_PSNR = PSNR_total / (len(file_list_div) - 1)

                # process continuous problem
                Threhold = 2 / 3

                # print('Thre: '+str(Threhold * Avg_PSNR))

                if i == 0:
                    last_frame = cv2.imread(os.path.join(folder_path, '00000.png'), cv2.IMREAD_UNCHANGED)
                    last_img_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
                    para = 0

                for indx, file in enumerate(file_list_div):

                    current_file_path = os.path.join(folder_path, file)
                    # next_name = '{:05d}.png'.format((int(file[:-4]) + 1))
                    # next_file_path = os.path.join(folder_path, next_name)

                    current_img = cv2.imread(current_file_path, cv2.IMREAD_UNCHANGED)  # HWC
                    # next_img = cv2.imread(next_file_path, cv2.IMREAD_UNCHANGED)

                    current_img_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

                    # next_img_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

                    # calculate the optical flow to determine if it is too normal sample

                    # current_img_gray = cv2.resize(current_img, tuple(resize_input))
                    # next_img_gray = cv2.resize(next_img_gray, tuple(resize_input))

                    psnr = calculate_psnr(current_img_gray, last_img_gray)

                    # print(file,next_name,'psnr',psnr)

                    if psnr < Threhold * Avg_PSNR:
                        para = para + 1

                    Noation[folder_].append(para)

                    # print(file, 'psnr', psnr, para)

                    last_img_gray = current_img_gray

            # print("Finish", folder_)
            if folder_index % 100 == 0:
                with open(os.path.join(OutputDir, '{}_{:04d}.json'.format(save_name, folder_index)), 'w') as outfile:
                    json.dump(Noation, outfile)

        with open(os.path.join(OutputDir, '{}'.format(save_name) + '.json'), 'w') as outfile:
            json.dump(Noation, outfile)
    else:

        import collections

        with open(os.path.join(OutputDir, '{}'.format(save_name) + '.json')) as f:
            data = json.load(f)

        LEN = len(data)

        print("Start Processing ...")
        sum = 0
        SUM = 0
        for folder in data:
            print(folder)
            folder_notation = data[folder]
            num = len(folder_notation)
            d = collections.Counter(folder_notation)
            SUM = SUM + num
            sum = sum + len(d)

        print("All screen change:", sum)
        print("Ratio, ", sum / SUM)


# print(data)


process(args.Input540pDir, args.OutputDir, args.save_name)
