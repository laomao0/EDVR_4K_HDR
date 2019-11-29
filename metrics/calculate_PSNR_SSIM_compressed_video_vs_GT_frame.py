'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob
import argparse
import os.path
import shutil

def rename_images(folder_path):

    images = sorted(os.listdir(folder_path))

    for img in images:
        img_path = os.path.join(folder_path, img)
        index = int(img[:-4]) - 1
        new_img = '{:05d}.png'.format(index)

        new_img_path = os.path.join(folder_path, new_img)

        os.rename(img_path, new_img_path)


def extract_frames(video, inDir, outDir, args):
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

    if os.path.exists(outDir):

        retn = os.system(
            '{} -i {}  {}/%05d.png'.format(
                os.path.join(args.ffmpeg_dir, "ffmpeg"),
                inDir,
                outDir))

        # retn = os.system(
        #     '{} -r 25 -i {}/%05d.png -codec copy -fs 60MB {}.mp4'.format(
        #         os.path.join(args.ffmpeg_dir, "ffmpeg"),
        #         inDir,
        #         outDir))

        rename_images(outDir)

        if retn:
            print("Error converting file:{}. Exiting.".format(video))
    else:

        print("Video existing !")

    print("Success extract {}".format(video))


def main():
    # Configurations

    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_dir", type=str, required=True, help='path to ffmpeg.exe')
    parser.add_argument("--gt_path", type=str, required=True, help='path to the folder containing gt images')
    parser.add_argument("--compressed_video_path", type=str, required=True, help='path to the folder containing compressed videos')
    args = parser.parse_args()

    # extract video to frames
    all_compressed_videos_tmp = os.listdir(args.compressed_video_path)
    all_compressed_videos = []
    for video in all_compressed_videos_tmp:
        if video.endswith('.mp4'):
            all_compressed_videos.append(video)

    #create tmp folders
    tmp_path = os.path.join(args.compressed_video_path, 'tmp')
    # tmp_path = os.path.join(args.compressed_video_path, '..', 'tmp')
    if os.path.exists(tmp_path):
        # os.removedirs(tmp_path)
        shutil.rmtree(tmp_path)

    os.mkdir(tmp_path)

    # flesh log
    log_path = os.path.join(args.compressed_video_path, "log.txt")
    if os.path.exists(os.path.join(log_path)):
        os.remove(log_path)


    print("We evaluate {} vidoes".format(len(all_compressed_videos)))



    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    # folder_GT = '/mnt/SSD/xtwang/BasicSR_datasets/val_set5/Set5'
    # folder_Gen = '/home/xtwang/Projects/BasicSR/results/RRDB_PSNR_x4/set5'

    # create tmp folder to store compressed frames.

    folder_GT = args.gt_path

    crop_border = 0
    print("crop_border: ", crop_border)

    suffix = ''  # suffix for Gen images
    test_Y = False  # True: test Y channel only; False: test RGB channels

    PSNR_all_folders = []
    SSIM_all_folders = []
    All_folders = []

    for folder in all_compressed_videos: # todo

        folder = folder[:-4]
        All_folders.append(folder)

        video_name = folder
        video_path = os.path.join(args.compressed_video_path, folder+'.mp4')
        output_path = os.path.join(tmp_path, video_name)
        os.mkdir(output_path)
        extract_frames(video_name, inDir=video_path, outDir=output_path, args=args)


        print("compute", folder)

        PSNR_all = []
        SSIM_all = []



        video_name_PATH = os.path.join(tmp_path, folder)

        img_list = sorted(os.listdir(video_name_PATH))

        if test_Y:
            print('Testing Y channel.')
        else:
            print('Testing RGB channels.')

        for i, name in enumerate(img_list):

            img_path = os.path.join(video_name_PATH, name)

            base_name = os.path.splitext(os.path.basename(img_path))[0]

            im_Gen  = cv2.imread(img_path) / 255.

            im_GT = cv2.imread(os.path.join(folder_GT, folder ,base_name + suffix + '.png')) / 255.

            if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
                im_GT_in = bgr2ycbcr(im_GT)
                im_Gen_in = bgr2ycbcr(im_Gen)
            else:
                im_GT_in = im_GT
                im_Gen_in = im_Gen

            # crop borders
            if im_GT_in.ndim == 3:
                if crop_border == 0:
                    cropped_GT = im_GT_in
                    cropped_Gen = im_Gen_in
                else:
                    cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
                    cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
            elif im_GT_in.ndim == 2:
                cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
                cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
            else:
                raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

            # calculate PSNR and SSIM
            PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)

            # SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
            SSIM = 0  #todo save time


            pstring = ('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
                i + 1, base_name, PSNR, SSIM))
            print(pstring)
            print(pstring, file=open(log_path, "a"))

            PSNR_all.append(PSNR)
            SSIM_all.append(SSIM)


        pstring = ('Video {}.mp4 Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
            folder,
            sum(PSNR_all) / len(PSNR_all),
            sum(SSIM_all) / len(SSIM_all)))
        print(pstring)
        print(pstring, file=open(log_path, "a"))

        PSNR_all_folders.append(sum(PSNR_all) / len(PSNR_all))
        SSIM_all_folders.append(sum(SSIM_all) / len(SSIM_all))


    print("--------------------------- all folder list here --------------")
    for index in range(len(PSNR_all_folders)):
        pstring = 'Forder {} Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format( All_folders[index],
                                                                             PSNR_all_folders[index],
                                                                             SSIM_all_folders[index])
        print(pstring)
        print(pstring, file=open(log_path, "a"))

    pstring = ('All Validation set Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
        sum(PSNR_all_folders) / len(PSNR_all_folders),
        sum(SSIM_all_folders) / len(SSIM_all_folders)))
    print(pstring)
    print(pstring, file=open(log_path, "a"))




    shutil.rmtree(tmp_path)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    main()
