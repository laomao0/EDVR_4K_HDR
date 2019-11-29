"""Create lmdb files for [General images (291 images/DIV2K) | Vimeo90K | REDS] training datasets

    Because the traing is too slow, We precropping the 4k imgae into 1020x1024 patches.

"""

import sys
import os.path as osp
import glob
import pickle
from multiprocessing import Pool
import numpy as np
import lmdb
import cv2

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import data.util as data_util  # noqa: E402
import utils.util as util  # noqa: E402
import argparse

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
                    help='vimeo90K | REDS | 4KHDR| general (e.g., DIV2K, 291) | DIV2K_demo |test')
parser.add_argument("--mode", type=str, required=True, help='4kHDR: train_4k, train_540p')
parser.add_argument("--name", type=str, required=True, help='4kHDR: train_4k, train_540p')
parser.add_argument("--small", type=int, required=True, help='Use samll set')
parser.add_argument("--batch", type=int, required=True, help='Use how many frames in a batch')
args = parser.parse_args()

Test = 0


def main():
    dataset = args.dataset  # vimeo90K | REDS | 4KHDR| general (e.g., DIV2K, 291) | DIV2K_demo |test
    mode = args.mode  # used for vimeo90k and REDS datasets
    # vimeo90k: GT | LR | flow
    # REDS: train_sharp, train_sharp_bicubic, train_blur_bicubic, train_blur, train_blur_comp
    #       train_sharp_flowx4
    # 4kHDR: train_4k, train_540p
    if dataset == 'vimeo90k':
        vimeo90k(mode)
    elif dataset == 'REDS':
        REDS(mode)
    elif dataset == '4KHDR':
        HDR(mode)
    elif dataset == 'general':
        opt = {}
        opt['img_folder'] = '../../datasets/DIV2K/DIV2K800_sub'
        opt['lmdb_save_path'] = '../../datasets/DIV2K/DIV2K800_sub.lmdb'
        opt['name'] = 'DIV2K800_sub_GT'
        general_image_folder(opt)
    elif dataset == 'DIV2K_demo':
        opt = {}
        ## GT
        opt['img_folder'] = '../../datasets/DIV2K/DIV2K800_sub'
        opt['lmdb_save_path'] = '../../datasets/DIV2K/DIV2K800_sub.lmdb'
        opt['name'] = 'DIV2K800_sub_GT'
        general_image_folder(opt)
        ## LR
        opt['img_folder'] = '../../datasets/DIV2K/DIV2K800_sub_bicLRx4'
        opt['lmdb_save_path'] = '../../datasets/DIV2K/DIV2K800_sub_bicLRx4.lmdb'
        opt['name'] = 'DIV2K800_sub_bicLRx4'
        general_image_folder(opt)
    elif dataset == 'test':
        test_lmdb('../../datasets/REDS/train_sharp_wval.lmdb', 'REDS')


def read_image_worker(path, key):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)


def general_image_folder(opt):
    """Create lmdb for general image folders
    Users should define the keys, such as: '0321_s035' for DIV2K sub-images
    If all the images have the same resolution, it will only store one copy of resolution info.
        Otherwise, it will store every resolution info.
    """
    #### configurations
    read_all_imgs = False  # whether real all images to memory with multiprocessing
    # Set False for use limited memory
    BATCH = 5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    n_thread = 40
    ########################################################
    img_folder = opt['img_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    all_img_list = sorted(glob.glob(osp.join(img_folder, '*')))
    keys = []
    for img_path in all_img_list:
        keys.append(osp.splitext(osp.basename(img_path))[0])

    if read_all_imgs:
        #### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            '''get the image data and update pbar'''
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    #### create lmdb environment
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    #### write data to lmdb
    pbar = util.ProgressBar(len(all_img_list))
    txn = env.begin(write=True)
    resolutions = []
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data.ndim == 2:
            H, W = data.shape
            C = 1
        else:
            H, W, C = data.shape
        txn.put(key_byte, data)
        resolutions.append('{:d}_{:d}_{:d}'.format(C, H, W))
        if not read_all_imgs and idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    # check whether all the images are the same size
    assert len(keys) == len(resolutions)
    if len(set(resolutions)) <= 1:
        meta_info['resolution'] = [resolutions[0]]
        meta_info['keys'] = keys
        print('All images have the same resolution. Simplify the meta info.')
    else:
        meta_info['resolution'] = resolutions
        meta_info['keys'] = keys
        print('Not all images have the same resolution. Save meta info for each image.')

    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def vimeo90k(mode):
    """Create lmdb for the Vimeo90K dataset, each image with a fixed size
    GT: [3, 256, 448]
        Now only need the 4th frame, e.g., 00001_0001_4
    LR: [3, 64, 112]
        1st - 7th frames, e.g., 00001_0001_1, ..., 00001_0001_7
    key:
        Use the folder and subfolder names, w/o the frame index, e.g., 00001_0001

    flow: downsampled flow: [3, 360, 320], keys: 00001_0001_4_[p3, p2, p1, n1, n2, n3]
        Each flow is calculated with GT images by PWCNet and then downsampled by 1/4
        Flow map is quantized by mmcv and saved in png format
    """
    #### configurations
    read_all_imgs = False  # whether real all images to memory with multiprocessing
    # Set False for use limited memory
    BATCH = 5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    if mode == 'GT':
        img_folder = '../../datasets/vimeo90k/vimeo_septuplet/sequences'
        lmdb_save_path = '../../datasets/vimeo90k/vimeo90k_train_GT.lmdb'
        txt_file = '../../datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
        H_dst, W_dst = 256, 448
    elif mode == 'LR':
        img_folder = '../../datasets/vimeo90k/vimeo_septuplet_matlabLRx4/sequences'
        lmdb_save_path = '../../datasets/vimeo90k/vimeo90k_train_LR7frames.lmdb'
        txt_file = '../../datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
        H_dst, W_dst = 64, 112
    elif mode == 'flow':
        img_folder = '../../datasets/vimeo90k/vimeo_septuplet/sequences_flowx4'
        lmdb_save_path = '../../datasets/vimeo90k/vimeo90k_train_flowx4.lmdb'
        txt_file = '../../datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
        H_dst, W_dst = 128, 112
    else:
        raise ValueError('Wrong dataset mode: {}'.format(mode))
    n_thread = 40
    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]
    all_img_list = []
    keys = []
    for line in train_l:
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        all_img_list.extend(glob.glob(osp.join(img_folder, folder, sub_folder, '*')))
        if mode == 'flow':
            for j in range(1, 4):
                keys.append('{}_{}_4_n{}'.format(folder, sub_folder, j))
                keys.append('{}_{}_4_p{}'.format(folder, sub_folder, j))
        else:
            for j in range(7):
                keys.append('{}_{}_{}'.format(folder, sub_folder, j + 1))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)
    if mode == 'GT':  # only read the 4th frame for the GT mode
        print('Only keep the 4th frame.')
        all_img_list = [v for v in all_img_list if v.endswith('im4.png')]
        keys = [v for v in keys if v.endswith('_4')]

    if read_all_imgs:
        #### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            """get the image data and update pbar"""
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    #### write data to lmdb
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    txn = env.begin(write=True)
    pbar = util.ProgressBar(len(all_img_list))
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if 'flow' in mode:
            H, W = data.shape
            assert H == H_dst and W == W_dst, 'different shape.'
        else:
            H, W, C = data.shape
            assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, data)
        if not read_all_imgs and idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    if mode == 'GT':
        meta_info['name'] = 'Vimeo90K_train_GT'
    elif mode == 'LR':
        meta_info['name'] = 'Vimeo90K_train_LR'
    elif mode == 'flow':
        meta_info['name'] = 'Vimeo90K_train_flowx4'
    channel = 1 if 'flow' in mode else 3
    meta_info['resolution'] = '{}_{}_{}'.format(channel, H_dst, W_dst)
    key_set = set()
    for key in keys:
        if mode == 'flow':
            a, b, _, _ = key.split('_')
        else:
            a, b, _ = key.split('_')
        key_set.add('{}_{}'.format(a, b))
    meta_info['keys'] = list(key_set)
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def REDS(mode):
    """Create lmdb for the REDS dataset, each image with a fixed size
    GT: [3, 720, 1280], key: 000_00000000
    LR: [3, 180, 320], key: 000_00000000
    key: 000_00000000

    flow: downsampled flow: [3, 360, 320], keys: 000_00000005_[p2, p1, n1, n2]
        Each flow is calculated with the GT images by PWCNet and then downsampled by 1/4
        Flow map is quantized by mmcv and saved in png format
    """
    #### configurations
    read_all_imgs = False  # whether real all images to memory with multiprocessing
    # Set False for use limited memory
    BATCH = 5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    if mode == 'train_sharp':
        # img_folder = '../../datasets/REDS/train_sharp'
        # lmdb_save_path = '../../datasets/REDS/train_sharp_wval.lmdb'
        img_folder = '/DATA/wangshen_data/REDS/train_sharp'
        lmdb_save_path = '/DATA/wangshen_data/REDS/train_sharp_wval.lmdb'
        H_dst, W_dst = 720, 1280
    elif mode == 'train_sharp_bicubic':
        # img_folder = '../../datasets/REDS/train_sharp_bicubic'
        # lmdb_save_path = '../../datasets/REDS/train_sharp_bicubic_wval.lmdb'
        img_folder = '/DATA/wangshen_data/REDS/train_sharp_bicubic'
        lmdb_save_path = '/DATA/wangshen_data/REDS/train_sharp_bicubic_wval.lmdb'
        H_dst, W_dst = 180, 320
    elif mode == 'train_blur_bicubic':
        img_folder = '../../datasets/REDS/train_blur_bicubic'
        lmdb_save_path = '../../datasets/REDS/train_blur_bicubic_wval.lmdb'
        H_dst, W_dst = 180, 320
    elif mode == 'train_blur':
        img_folder = '../../datasets/REDS/train_blur'
        lmdb_save_path = '../../datasets/REDS/train_blur_wval.lmdb'
        H_dst, W_dst = 720, 1280
    elif mode == 'train_blur_comp':
        img_folder = '../../datasets/REDS/train_blur_comp'
        lmdb_save_path = '../../datasets/REDS/train_blur_comp_wval.lmdb'
        H_dst, W_dst = 720, 1280
    elif mode == 'train_sharp_flowx4':
        img_folder = '../../datasets/REDS/train_sharp_flowx4'
        lmdb_save_path = '../../datasets/REDS/train_sharp_flowx4.lmdb'
        H_dst, W_dst = 360, 320
    n_thread = 40
    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    all_img_list = data_util._get_paths_from_images(img_folder)
    keys = []
    for img_path in all_img_list:
        split_rlt = img_path.split('/')
        folder = split_rlt[-2]
        img_name = split_rlt[-1].split('.png')[0]
        keys.append(folder + '_' + img_name)

    if read_all_imgs:
        #### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            '''get the image data and update pbar'''
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    #### create lmdb environment
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    #### write data to lmdb
    pbar = util.ProgressBar(len(all_img_list))
    txn = env.begin(write=True)
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if 'flow' in mode:
            H, W = data.shape
            assert H == H_dst and W == W_dst, 'different shape.'
        else:
            H, W, C = data.shape
            assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, data)
        if not read_all_imgs and idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    meta_info['name'] = 'REDS_{}_wval'.format(mode)
    channel = 1 if 'flow' in mode else 3
    meta_info['resolution'] = '{}_{}_{}'.format(channel, H_dst, W_dst)
    meta_info['keys'] = keys
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def HDR(mode):
    """Create lmdb for the REDS dataset, each image with a fixed size
    GT: [3, 720, 1280], key: 000_00000000
    LR: [3, 180, 320], key: 000_00000000
    key: 000_00000000

    flow: downsampled flow: [3, 360, 320], keys: 000_00000005_[p2, p1, n1, n2]
        Each flow is calculated with the GT images by PWCNet and then downsampled by 1/4
        Flow map is quantized by mmcv and saved in png format
    """
    #### configurations
    read_all_imgs = False  # whether real all images to memory with multiprocessing
    # Set False for use limited memory
    BATCH = args.batch  # After BATCH images, lmdb commits, if read_all_imgs = False
    if mode == 'train_sharp':
        # img_folder = '../../datasets/REDS/train_sharp'
        # lmdb_save_path = '../../datasets/REDS/train_sharp_wval.lmdb'
        img_folder = '/DATA/wangshen_data/REDS/train_sharp'
        lmdb_save_path = '/DATA/wangshen_data/REDS/train_sharp_wval.lmdb'
        H_dst, W_dst = 720, 1280
    elif mode == 'train_sharp_bicubic':
        # img_folder = '../../datasets/REDS/train_sharp_bicubic'
        # lmdb_save_path = '../../datasets/REDS/train_sharp_bicubic_wval.lmdb'
        img_folder = '/DATA/wangshen_data/REDS/train_sharp_bicubic'
        lmdb_save_path = '/DATA/wangshen_data/REDS/train_sharp_bicubic_wval.lmdb'
        H_dst, W_dst = 180, 320
    elif mode == 'train_blur_bicubic':
        img_folder = '../../datasets/REDS/train_blur_bicubic'
        lmdb_save_path = '../../datasets/REDS/train_blur_bicubic_wval.lmdb'
        H_dst, W_dst = 180, 320
    elif mode == 'train_blur':
        img_folder = '../../datasets/REDS/train_blur'
        lmdb_save_path = '../../datasets/REDS/train_blur_wval.lmdb'
        H_dst, W_dst = 720, 1280
    elif mode == 'train_blur_comp':
        img_folder = '../../datasets/REDS/train_blur_comp'
        lmdb_save_path = '../../datasets/REDS/train_blur_comp_wval.lmdb'
        H_dst, W_dst = 720, 1280
    elif mode == 'train_sharp_flowx4':
        img_folder = '../../datasets/REDS/train_sharp_flowx4'
        lmdb_save_path = '../../datasets/REDS/train_sharp_flowx4.lmdb'
        H_dst, W_dst = 360, 320
    elif mode == 'train_540p':
        img_folder = "/mnt/lustre/shanghai/cmic/home/xyz18/Dataset/train_540p"
        lmdb_save_path = '/mnt/lustre/shanghai/cmic/home/xyz18/Dataset/{}.lmdb'.format(args.name)
        H_dst, W_dst = 540, 960
    elif mode == 'train_4k':
        img_folder = "/mnt/lustre/shanghai/cmic/home/xyz18/Dataset/train_4k"
        lmdb_save_path = '/mnt/lustre/shanghai/cmic/home/xyz18/Dataset/{}.lmdb'.format(args.name)
        H_dst, W_dst = 540, 960
        H_dst, W_dst = 2160, 3840
    elif mode == 'both':
        img_folder_S = "/mnt/lustre/shanghai/cmic/home/xyz18/Dataset/train_540p"
        lmdb_save_path_S = '/mnt/lustre/shanghai/cmic/home/xyz18/Dataset/{}_540p.lmdb'.format(args.name)
        H_dst_S, W_dst_S = 256, 256
        img_folder_L = "/mnt/lustre/shanghai/cmic/home/xyz18/Dataset/train_4k"
        lmdb_save_path_L = '/mnt/lustre/shanghai/cmic/home/xyz18/Dataset/{}_4k.lmdb'.format(args.name)
        H_dst_L, W_dst_L = 1024, 1024

    assert mode == 'both'

    N = 8  # divide one 4k into 8 parts

    n_thread = 40


    ########################################################

    import os
    import shutil
    if os.path.exists(lmdb_save_path_S):
        shutil.rmtree(lmdb_save_path_S)
    if os.path.exists(lmdb_save_path_L):
        shutil.rmtree(lmdb_save_path_L)

    if not lmdb_save_path_S.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path_S):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path_S))
        sys.exit(1)

    if not lmdb_save_path_L.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path_L):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path_L))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    all_img_list_S = data_util._get_paths_from_images_suzhou(img_folder_S, args.small)
    all_img_list_L = data_util._get_paths_from_images_suzhou(img_folder_L, args.small)

    keys_S = []
    keys_L = []
    for img_path in all_img_list_S:
        split_rlt = img_path.split('/')
        folder = split_rlt[-2]
        img_name = split_rlt[-1].split('.png')[0]
        keys_S.append(folder + '_' + img_name)

    for img_path in all_img_list_L:
        split_rlt = img_path.split('/')
        folder = split_rlt[-2]
        img_name = split_rlt[-1].split('.png')[0]
        keys_L.append(folder + '_' + img_name)

    assert keys_S == keys_L

    keys_patch = []
    for img_path in all_img_list_L:
        split_rlt = img_path.split('/')
        folder = split_rlt[-2]
        img_name = split_rlt[-1].split('.png')[0]
        for i in range(N):
            keys_patch.append(folder + '_' + img_name + '_' + str(i))

    keys = keys_S

    if read_all_imgs:  #  todo never use that
        # read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list_S))

        def mycallback(arg):
            '''get the image data and update pbar'''
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list_S, keys_S):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list_S)))

    #### create lmdb environment


    # for small pic
    data_size_per_img = cv2.imread(all_img_list_S[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per small image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list_S)
    # env_S = lmdb.open(lmdb_save_path_S, map_size=data_size * 10)

    # for large pic
    data_size_per_img = cv2.imread(all_img_list_L[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per large image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list_S)
    # env_L = lmdb.open(lmdb_save_path_L, map_size=data_size * 10)




    #### write data to lmdb
    pbar = util.ProgressBar(len(all_img_list_S))

    # txn_S = env_S.begin(write=True)
    # txn_L = env_L.begin(write=True)

    for idx_all, (path_S, path_L, key) in enumerate(zip(all_img_list_S, all_img_list_L, keys)):

        # pbar.update('Write {}'.format(key))

        data_S = dataset[key] if read_all_imgs else cv2.imread(path_S, cv2.IMREAD_UNCHANGED)  # shape H W C  ndarray
        data_L = dataset[key] if read_all_imgs else cv2.imread(path_L, cv2.IMREAD_UNCHANGED)

        # process the black blank
        H_S = data_S.shape[0]  # 540
        W_S = data_S.shape[1]  # 960
        H_L = data_L.shape[0]
        W_L = data_L.shape[1]

        blank_1_S = 0
        blank_2_S = 0
        for i in range(H_S):
            if not sum(data_S[:,:,0][i]) == 0:
                blank_1_S = i - 1
                # assert not sum(data_S[:, :, 0][i+1]) == 0
                break

        for i in range(H_S):
            if not sum(data_S[:, :, 0][H_S - i - 1 ]) == 0:
                blank_2_S = (H_S - 1) - i - 1
                # assert not sum(data_S[:, :, 0][blank_2_S-1]) == 0
                break
        print('LQ :', blank_1_S, blank_2_S)

        if blank_1_S == -1:
            print('LQ has no blank')

        blank_1_L = 0
        blank_2_L = 0
        for i in range(H_L):
            if not sum(data_L[:, :, 0][i]) == 0:
                blank_1_L = i - 1
                # assert not sum(data_L[:, :, 0][i + 1]) == 0
                break

        for i in range(H_L):
            if not sum(data_L[:, :, 0][H_L - i - 1]) == 0:
                blank_2_L = (H_L - 1) - i - 1
                # assert not sum(data_L[:, :, 0][blank_2_L - 1]) == 0
                break
        print('GT :', blank_1_L, blank_2_L)
        if blank_1_L == -1 and blank_2_L == H_L-2:
            print('No blank', key)
            U_L1 = 56
            D_L2 = 2104
        else:
            U_L1 = ((blank_1_L >> 2) + 1) << 2
            D_L2 = ((blank_2_L >> 2) - 1) << 2

        print('Content:', U_L1, D_L2)

        # crop into eight patches
        U_L1_d = U_L1 + H_dst_L
        D_L2_u = D_L2 - H_dst_L

        assert U_L1_d <= H_L
        assert D_L2_u >= 0

        # for idx in range(N):



        # crop eight part
        H_list = [U_L1, D_L2_u]
        W_list = [0, 1024, 2048, 2816]
        h_list = [U_L1//4,D_L2_u//4]
        w_list = [0, 256, 512, 704]

        for h_idx, _ in enumerate(H_list):
            for w_idx, _ in enumerate(W_list):
                key_idx = key + '_' + str(4*h_idx + w_idx)
                print(key_idx)
                key_byte = key_idx.encode('ascii')
                data_gt = data_L[H_list[h_idx]:(H_list[h_idx]+1024), W_list[w_idx]:(W_list[w_idx]+1024),:]
                data_lq = data_S[h_list[h_idx]:(h_list[h_idx]+256), w_list[w_idx]:(w_list[w_idx]+256),:]
                # txn_L.put(key_byte, data_gt.copy(order='C'))
                # txn_S.put(key_byte, data_lq.copy(order='C'))


        # if not read_all_imgs and idx_all % BATCH == 0:
            # txn_L.commit()
            # txn_S.commit()
            # txn_L = env_L.begin(write=True)
            # txn_S = env_S.begin(write=True)


    # txn_L.commit()
    # txn_S.commit()
    # env_L.close()
    # env_S.close()
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    meta_info['name'] = 'HDR_{}_wval'.format(mode)
    channel = 1 if 'flow' in mode else 3
    meta_info['resolution'] = '{}_{}_{}'.format(channel, H_dst_L, W_dst_L)
    meta_info['keys'] = keys_patch
    pickle.dump(meta_info, open(osp.join(lmdb_save_path_L, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

    # for LQ
    meta_info = {}
    meta_info['name'] = 'HDR_{}_wval'.format(mode)
    channel = 1 if 'flow' in mode else 3
    meta_info['resolution'] = '{}_{}_{}'.format(channel, H_dst_S, W_dst_S)
    meta_info['keys'] = keys_patch
    pickle.dump(meta_info, open(osp.join(lmdb_save_path_S, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def test_lmdb(dataroot, dataset='REDS'):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(osp.join(dataroot, 'meta_info.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))
    # read one image
    if dataset == 'vimeo90k':
        key = '00001_0001_4'
    else:
        key = '00000_00042'

    folders = ['11553969', '10306321', '12356337']

    # for i in range(10):
    for i in folders:
        for j in range(1, 100):
            for h in range(8):
                key = '{}_{:05d}_{:01d}'.format(i, j, h)
                print('Reading {} for test.'.format(key))
                with env.begin(write=False) as txn:
                    buf = txn.get(key.encode('ascii'))
                img_flat = np.frombuffer(buf, dtype=np.uint8)
                C, H, W = [int(s) for s in meta_info['resolution'].split('_')]
                img = img_flat.reshape(H, W, C)
                cv2.imwrite('/mnt/lustre/shanghai/cmic/home/xyz18/Dataset/test/{}.png'.format(key), img)


if __name__ == "__main__":
    if Test == 1:
        test_lmdb('/mnt/lustre/shanghai/cmic/home/xyz18/Dataset/train_precropping_4k.lmdb')
    else:
        main()
