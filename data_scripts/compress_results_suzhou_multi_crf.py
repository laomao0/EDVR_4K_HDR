import argparse
import os
import os.path
from shutil import rmtree, move, copy
import random
from scipy.ndimage import imread
from scipy.misc import imsave
import shutil
import math

exceed_list = ['65342200', '39167513']

def dir_file_size(path):
    if os.path.isdir(path):
        file_size = 0
        dir_list = os.listdir(path)
        for dir_name in dir_list:
            file_path = os.path.join(path, dir_name)
            if os.path.isfile(dir_name):
                file_size += os.path.getsize(file_path)
            else:
                ret = dir_file_size(file_path)
                file_size += ret
        return file_size
    elif os.path.isfile(path):
        return os.path.getsize(path)
    else:
        print('找不到%s文件' % path)

def combine_frames(video, inDir, outDir, args, crf):
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

    if not os.path.exists(outDir+'.mp4'):

        # retn = os.system(
        #     '{} -r 25 -i {}/%05d.png -b:v {}M -vcodec libx265 {}.mp4'.format(
        #         os.path.join(args.c, "ffmpeg"),
        #         inDir,
        #         str(args.bitrate),
        #         outDir))
        print((
            '{} -r 24000/1001 -i {}/%05d.png -vcodec libx265 -pix_fmt yuv422p -crf {} {}.mp4'.format(
                args.ffmpeg_dir,
                inDir,
                crf,
                outDir)))


        retn = os.system(
            '{} -r 24000/1001 -i {}/%05d.png -vcodec libx265 -pix_fmt yuv422p -crf {} {}.mp4'.format(
                args.ffmpeg_dir,
                inDir,
                crf,
                outDir))


        # retn = os.system(
        #     '{} -r 25 -i {}/%05d.png -codec copy -fs 60MB {}.mp4'.format(
        #         os.path.join(args.ffmpeg_dir, "ffmpeg"),
        #         inDir,
        #         outDir))

        if retn:
            print("Error converting file:{}. Exiting.".format(video))
    else:

        print("Video existing !")

    print("Success output {}.mp4".format(video))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_dir", type=str, required=True, help='path to ffmpeg.exe')
    parser.add_argument("--imgs_folder_input", type=str, required=True, help='path to the folder containing images')
    parser.add_argument("--output_folder", type=str, required=True, help='path to the output dataset folder')
    parser.add_argument("--crf", type=str, required=True, help='codec bitrate')
    args = parser.parse_args()

    tmp_all_img_folders = os.listdir(args.imgs_folder_input)

    all_img_folders = []

    # exclude log file
    for folder in tmp_all_img_folders:
        if not folder.endswith('.log'):
            all_img_folders.append(folder)

    all_img_folders = sorted(all_img_folders)

    # create video saving path
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    for folder in all_img_folders:
        folder_path = os.path.join(args.imgs_folder_input, folder)
        imgs = sorted(os.listdir(folder_path))
        length = len(imgs)
        assert length == 100
        assert imgs[0] == '00000.png'

        # path for output video
        output_video_path = os.path.join(args.output_folder, folder)

        folder_size = dir_file_size(folder_path) / 1024 / 1024
        print ('folder_size: ', folder_size)

        crf = folder_size // 100
        if (folder in exceed_list):
            crf += 1

        combine_frames(video=folder, inDir=folder_path, outDir=output_video_path, args=args, crf = crf)


if __name__ == "__main__":
    main()
