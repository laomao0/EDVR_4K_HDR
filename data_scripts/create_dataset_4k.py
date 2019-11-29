"""
    This script generates a motion blur sequences by moving average a 9 frame length window.
    The blurry sequences is 30fps.
    Take a example, the original video is 240-fps.
    240-fps Sequence: 0 1 2 3 4 5 6 7 8 9 10 ...
    We select 8  16  24... as the center output blurry index, for index 8, we averge [8-(9-1)/2, 8+(9-1)/2],
    i.e. [4, 12] averaged to form the index 8 blurry frame.
"""

import argparse
import os
import os.path
from shutil import rmtree, move, copy
import random
from scipy.ndimage import imread
from scipy.misc import imsave
import shutil
import math

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, required=True, help='path to ffmpeg.exe')
parser.add_argument("--dataset", type=str, default="adobe240fps_blur", help='specify if using "adobe240fps" or custom video dataset')
parser.add_argument("--train_4k_video_path", type=str, required=True, help='path to the folder containing 4k train videos')
parser.add_argument("--train_540p_video_path", type=str, required=True, help='path to the folder containing 540p test videos')
parser.add_argument("--test_540p_video_path", type=str, required=True, help='path to the folder containing 540p test videos')
parser.add_argument("--dataset_folder", type=str, required=True, help='path to the output dataset folder')
parser.add_argument("--img_width_gt", type=int, default=256, help="output image width")
parser.add_argument("--img_height_gt", type=int, default=128, help="output image height")
parser.add_argument("--img_width_input", type=int, default=256, help="output image width")
parser.add_argument("--img_height_input", type=int, default=128, help="output image height")
parser.add_argument("--train_test_split", type=tuple, default=(90, 10), help="train test split for custom dataset")
parser.add_argument("--window_size", type=int, default=7, help="number of frames to de average")
parser.add_argument("--enable_4k", default=0, type=int, help="generate train 4k data or not")
parser.add_argument("--enable_540p", default=0, type=int, help="generate train 540p data or not")
parser.add_argument("--enable_540p_test", default=0, type=int, help="generate train 540p test data or not")
args = parser.parse_args()

debug = False
delte_extract = False
#print(args)

def rename_images(folder_path):

    images = sorted(os.listdir(folder_path))

    for img in images:
        img_path = os.path.join(folder_path, img)
        index = int(img[:-4]) - 1
        new_img = '{:05d}.png'.format(index)

        new_img_path = os.path.join(folder_path, new_img)

        os.rename(img_path, new_img_path)


def extract_frames(videos, inDir, outDir, width=0, height=0):
    """
    Converts all the videos passed in `videos` list to images.

    Parameters
    ----------
        videos : list
            name of all video files.
        inDir : string
            path to input directory containing videos in `videos` list.
        outDir : string
            path to directory to output the extracted images.

    Returns
    -------
        None
    """

    for video in videos:

        if not os.path.exists(os.path.join(outDir, os.path.splitext(video)[0])):

            os.makedirs(os.path.join(outDir, os.path.splitext(video)[0]), exist_ok=True)
            # retn = os.system(
            #     '{} -i {} -vf scale={}:{} -vsync 0 -qscale:v 2 {}/%05d.png'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"),
            #                                                                        os.path.join(inDir, video),
            #                                                                        width, height,
            #                                                                        os.path.join(outDir,
            #                                                                                     os.path.splitext(video)[
            #                                                                                         0])))
            retn = os.system(
                '{} -i {} {}/%05d.png'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"),
                                                os.path.join(inDir, video),
                                              os.path.join(outDir, os.path.splitext(video)[0]) ))

            rename_images(os.path.join(outDir, os.path.splitext(video)[0]))

            if retn:
                print("Error converting file:{}. Exiting.".format(video))

def main():
    # Create dataset folder if it doesn't exist already.
    if not os.path.isdir(args.dataset_folder):
        os.makedirs(args.dataset_folder, exist_ok=True)

    # extractPath = os.path.join(args.dataset_folder, "full_sharp")

    if args.enable_540p == 1:
        trainPath540p = os.path.join(args.dataset_folder, "train_540p")
        if not os.path.exists(trainPath540p):
            os.makedirs(trainPath540p, exist_ok=True)

    if args.enable_4k == 1:
        trainPath = os.path.join(args.dataset_folder, "train_4k")
        if not os.path.exists(trainPath):
            os.makedirs(trainPath, exist_ok=True)

    if args.enable_540p_test == 1:
        testPath540p = os.path.join(args.dataset_folder, "test_540p")
        if not os.path.exists(testPath540p):
            os.makedirs(testPath540p, exist_ok=True)



    if (args.dataset == "hdr" or args.dataset == "youtube240fps_blur"):

        if args.enable_540p == 1:
            videos = os.listdir(args.train_540p_video_path)
            for video in videos:
                extract_frames([video], args.train_540p_video_path, trainPath540p, width=args.img_width_input,
                               height=args.img_height_input)

        if args.enable_4k == 1:
            videos = os.listdir(args.train_4k_video_path)
            for video in videos:
                extract_frames([video], args.train_4k_video_path, trainPath, width=args.img_width_gt,
                               height=args.img_height_gt)

        if args.enable_540p_test == 1:
            videos = os.listdir(args.test_540p_video_path)
            for video in videos:
                extract_frames([video], args.test_540p_video_path, testPath540p, width=args.img_width_input,
                               height=args.img_height_input)









main()
