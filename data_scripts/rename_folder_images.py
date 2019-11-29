import os
import glob


def main():
    folder4k = '/DATA7_DB7/data/4khdr/data/Dataset/train_4k'
    folder540p = '/DATA7_DB7/data/4khdr/data/Dataset/train_540p'

    folders = sorted(os.listdir(folder4k))
    folders_540p = sorted(os.listdir(folder540p))
    len_folders = len(folders)

    for i,folder in enumerate(folders):

        print(folder)
        path4k = os.path.join(folder4k, folder)
        path540p = os.path.join(folder540p, folder)

        images = sorted(os.listdir(path540p))
        for img in images:
            img_path_4k = os.path.join(path4k, img)
            img_path_540p = os.path.join(path540p, img)
            index = int(img[:-4]) - 1
            new_img = '{:05d}.png'.format(index)

            new_img_path_4k = os.path.join(path4k, new_img)
            new_img_path_540p = os.path.join(path540p, new_img)

            os.rename(img_path_4k, new_img_path_4k)
            os.rename(img_path_540p, new_img_path_540p)


    print('Finished.')


def DIV2K(path):
    img_path_l = glob.glob(os.path.join(path, '*'))
    for img_path in img_path_l:
        new_path = img_path.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')
        os.rename(img_path, new_path)


if __name__ == "__main__":
    main()