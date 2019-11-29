import cv2
import os

video_path = '/mnt/lustre/shanghai/cmic/home/xyz18/raw/SDR_4k'

if __name__ == '__main__' :

    files = os.listdir(video_path)

    for item in files:
        item_path = os.path.join(video_path, item)

        video = cv2.VideoCapture(item_path)

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver)  < 3 :
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else :
            fps = video.get(cv2.CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        video.release()

