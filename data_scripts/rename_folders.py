import os
import glob


def main():
    folder4k = '/DATA7_DB7/data/4khdr/data/Dataset/train_4k'
    folder540p = '/DATA7_DB7/data/4khdr/data/Dataset/train_540p'

    folders = sorted(os.listdir(folder4k))
    folders_540p = sorted(os.listdir(folder540p))
    len_folders = len(folders)

    for i,folder in enumerate(folders):
        path = os.path.join(folder4k, folder)
        name = '{:05d}'.format(i)
        new_path = os.path.join(folder4k, name)
        os.rename(path, new_path)

    for i,folder in enumerate(folders_540p):
        path = os.path.join(folder540p, folder)
        name = '{:05d}'.format(i)
        new_path = os.path.join(folder540p, name)
        os.rename(path, new_path)




    print('Finished.')


def DIV2K(path):
    img_path_l = glob.glob(os.path.join(path, '*'))
    for img_path in img_path_l:
        new_path = img_path.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')
        os.rename(img_path, new_path)


if __name__ == "__main__":
    main()