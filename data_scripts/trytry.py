import os

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


path = '/mnt/lustre/shanghai/cmic/home/xyz18/Results/train_EDVR_L_Preload_30000_nobug_ssim_9_LR_1e_5_suzhou_remove_baddata/sharp_bicubic/98472411'
ret = dir_file_size(path)
print('{0} 的大小为 {1}字节'.format(path, ret))