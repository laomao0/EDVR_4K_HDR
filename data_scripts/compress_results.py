import argparse
import os
import os.path
from shutil import rmtree, move, copy
import random
from scipy.ndimage import imread
from scipy.misc import imsave
import shutil
import math


def combine_frames(video, inDir, outDir, args):
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
            '{} -r 24000/1001 -i {}/%05d.png -vcodec libx265  -pix_fmt yuv422p -crf10 {}.mp4'.format(
                "ffmpeg",
                inDir,
                outDir)))


        retn = os.system(
            'ffmpeg -r 24000/1001 -i {}/%05d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 {}.mp4'.format(
                inDir,
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
    parser.add_argument("--bitrate", type=str, required=True, help='codec bitrate')
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

        combine_frames(video=folder, inDir=folder_path, outDir=output_video_path, args=args)


if __name__ == "__main__":
    main()
