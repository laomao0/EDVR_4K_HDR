'''
REDS dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import json
import torch.utils.data as data
import data.util as util
try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger('base')


class REDSDataset(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt):
        super(REDSDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        #### directly load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
            logger.info('Using lmdb meta info for cache keys.')
        elif opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
        else:
            raise ValueError(
                'Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')

        # # remove the REDS4 for testing
        # self.paths_GT = [
        #     v for v in self.paths_GT if v.split('_')[0] not in ['000', '011', '015', '020']
        # ]

        # remove the 4Khdr for testing
        if not opt['all_dataset_for_train'] == True:
            print("exclude validation set for training")
            self.paths_GT = [
                v for v in self.paths_GT if v.split('_')[0] not in ['00000', '00001', '00002', '00003', '00004',
                                                                    '00005', '00006', '00007', '00008', '00009']
            ]

        if len(opt['baddata_list']) > 0:
            print("exclude bad training data")
            print(opt['baddata_list'])
            self.paths_GT = [
                v for v in self.paths_GT if v.split('_')[0] not in opt['baddata_list']
            ]


        print("Include validation set for training")

        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

        # load screen notation
        with open(opt['frame_notation']) as f:
            self.frame_notation = json.load(f)

        print("Training Dataset Initialized")

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        ''' Read BGR channels separately and then combine for 1M limits in cluster'''
        img_B = self._read_img_mc(osp.join(path + '_B', name_a, name_b + '.png'))
        img_G = self._read_img_mc(osp.join(path + '_G', name_a, name_b + '.png'))
        img_R = self._read_img_mc(osp.join(path + '_R', name_a, name_b + '.png'))
        img = cv2.merge((img_B, img_G, img_R))
        return img

    def check_notation(self, folder_key, center_index):
        """
        check not get dis-continous frames
        """
        notation = self.frame_notation[folder_key][center_index-2:center_index+3]
        if all(item == notation[0] for item in notation):
            return 1
        else:
            return 0

    def __getitem__(self, index):
        if self.data_type == 'mc':
            self._ensure_memcached()
        elif self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()

        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        name_a, name_b = key.split('_') # folder + img
        center_frame_idx = int(name_b)



        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            N_frames = self.opt['N_frames']
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) > 99:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 0:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval))
            name_b = '{:05d}'.format(neighbor_list[0])
        else:
            # ensure not exceeding the borders
            # ensure not samples the screen change


            while (center_frame_idx + self.half_N_frames * interval > 99) \
                    or (center_frame_idx - self.half_N_frames * interval < 0
                    or self.check_notation(name_a, center_frame_idx) == 0):  # check notation shenwang
                center_frame_idx = random.randint(0, 99)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            name_b = '{:05d}'.format(neighbor_list[self.half_N_frames])
            key = name_a + '_' + name_b #todo
        assert len(
            neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))

        # todo
        # print("GT_path", name_a, '_', name_b)



        #### get the GT image (as the center frame)
        if self.data_type == 'mc':
            img_GT = self._read_img_mc_BGR(self.GT_root, name_a, name_b)
            img_GT = img_GT.astype(np.float32) / 255.
        elif self.data_type == 'lmdb':
            # img_GT = util.read_img(self.GT_env, key, (3, 720, 1280))  # todo: update key
            img_GT = util.read_img(self.GT_env, key, (3, 2160, 3840))
        else:
            img_GT = util.read_img(None, osp.join(self.GT_root, name_a, name_b + '.png'))

        #### get LQ images
        # LQ_size_tuple = (3, 180, 320) if self.LR_input else (3, 720, 1280)
        LQ_size_tuple = (3, 540, 960) if self.LR_input else (3, 2160, 3840)
        img_LQ_l = []
        for v in neighbor_list:
            img_LQ_path = osp.join(self.LQ_root, name_a, '{:05d}.png'.format(v))

            # todo
            # print('LQ_path', img_LQ_path)

            if self.data_type == 'mc':
                if self.LR_input:
                    img_LQ = self._read_img_mc(img_LQ_path)
                else:
                    img_LQ = self._read_img_mc_BGR(self.LQ_root, name_a, '{:05d}'.format(v))
                img_LQ = img_LQ.astype(np.float32) / 255.
            elif self.data_type == 'lmdb':
                img_LQ = util.read_img(self.LQ_env, '{}_{:05d}'.format(name_a, v), LQ_size_tuple)
            else:
                img_LQ = util.read_img(None, img_LQ_path)
            img_LQ_l.append(img_LQ)
        assert key == name_a + '_' + name_b
        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale

                if self.opt['ignore_blank'] == True:
                    # print('ignore blank')
                    rnd_h = random.randint(80, max(80, H - LQ_size - 80))
                    rnd_w = random.randint(80, max(80, W - LQ_size - 80))

                    img_LQ_l_b = img_LQ_l[2][rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]

                    max_n = 10
                    n = 0
                    while ( np.mean(img_LQ_l_b) <= 10/255 and n <= max_n ):
                        n += 1
                        rnd_h = random.randint(80, max(80, H - LQ_size - 80))
                        rnd_w = random.randint(80, max(80, W - LQ_size - 80))
                        img_LQ_l_b = img_LQ_l[2][rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]

                else:
                    rnd_h = random.randint(0, max(0, H - LQ_size))
                    rnd_w = random.randint(0, max(0, W - LQ_size))
                    # print(rnd_h)
                    if rnd_h < 65:
                        print('stop')

                img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
                rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            else:
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_LQ_l.append(img_GT)
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GT, 'key': key}

    def __len__(self):
        return len(self.paths_GT)
