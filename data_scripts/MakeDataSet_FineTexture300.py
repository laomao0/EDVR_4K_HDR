####
# this is to make a dataset, including training, validation and test set for our Deep Learning model for Frame INterpolation
#
# Generally, i decided  to manage the dataset using TFRecord due to its full io read/write support for Tensorflow.
#
# And since we have to use the videos as a source for the model, the python-opencv packages is required.
# Moreover, to make the dataset more persuading, we choose the widely used Human Action Recognition UCF-101 Dataset as a source.

####

import cv2
import time
import sys
import os
import random
import numpy
import scipy
import scipy.stats
# import tensorflow as tf
from scipy.misc import imread, imsave, imshow, imresize, imsave
import matplotlib.pyplot as plt
# import numpy as np


TRAIN_PERCENTAGE = 90
VALIDATE_PERCENTAGE = 5
TEST_PERCENTAGE = 5

# tf_training_writer = tf.python_io.TFRecordWriter("FI_TrainigSet")
# tf_validate_writer = tf.python_io.TFRecordWriter("FI_ValidateSet")
# tf_test_writer = tf.python_io.TFRecordWriter("FI_TestSet")

# Make the output through Images.....
OutputDir = "/DATA5_DB8/data/4khdr/data/Dataset/"

# since there are many different action types and UCF-101 put similar action into a subdir
# i decided to choose
# VideoDir = "I:/UCF101HumanActionRecognitionVideo/UCF-101/"
VideoDir = "/DATA5_DB8/data/4khdr/data/SDR_4k/"
VideoDirInput = "/DATA5_DB8/data/4khdr/data/SDR_540p/"
# scale to 720p. select the former 10 seconds clip.
# this will be about 1000x10x30x 10 random pixel location
# thus we have about 3 Million samples
# each sample is composed of 5 image patches
SAMPLE_COUNT = 20000
dsize = [3840, 2160]
resize_input = [960, 540]

# sample_size = [300, 300]

sample_size = dsize

sample_count = 0


THRESHOLD_MOTION = 0.5
THRESHOLD_MOTION_HIGH = 40
THRESHOLD_TEXTURE_DIFF = 1.0/255

# gt_or_input = 1

if os.path.exists(os.path.join(OutputDir, "motionHistogram.txt")):
    motionHist = numpy.loadtxt(os.path.join(OutputDir, "motionHistogram.txt"),dtype=float,delimiter=",")
    motionHistsum = numpy.sum(motionHist[:,0])
else:
    bins  = int((THRESHOLD_MOTION_HIGH - THRESHOLD_MOTION)/0.5)+1
    motion = numpy.linspace(0, THRESHOLD_MOTION_HIGH, bins)
    # motionHist = numpy.zeros((THRESHOLD_MOTION_HIGH - THRESHOLD_MOTION)/0.5+1,numpy.int32)
    motionHist = numpy.zeros((bins,2),numpy.float32)
    motionHist[:,1] = motion
    numpy.savetxt("motionHistogram.txt",motionHist, fmt='%.2f', delimiter=",")

threshold = (motionHist[:,1]) / 16 * (1 - 1 / 16.0) + 1 / 16.0
counter = numpy.zeros_like(threshold)
COUNTMAX = numpy.round((numpy.divide(1,threshold)))
COUNTMAX = [1.0 if x==0 else x for x in COUNTMAX]
# threashold =

if os.path.exists(os.path.join(OutputDir,"im_list_rgb.txt")):
    fd = open(os.path.join(OutputDir,"im_list_rgb.txt"), "r")
    im_list_rgb = []
    im_list_sum = 0
    for line in fd.readlines():
        im_list_rgb.append(line)
        im_list_sum+=1
    fl = open("im_list_rgb.txt", 'w')
    sep = '\n'
    fl.write(sep.join(im_list_rgb))
    fl.close()

    # check
    if im_list_sum == motionHistsum:
        print("Check Correct")
        sample_count = im_list_sum
    else:
        print("Check failure")
        exit()
else:
    im_list_rgb = []



file_list = os.listdir(VideoDir)



group_num =  100
# group_sample =  40


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
videoid = 0
for file in file_list:
    i = 0
    print(file)
    if os.path.isdir(file):
        continue

    cap = cv2.VideoCapture(VideoDir + file)


    if not cap.isOpened():
        continue

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    if not os.path.exists(os.path.join(OutputDir, file[:-4])):
        os.mkdir(os.path.join(OutputDir, file[:-4]))
    else:
        videoid +=1
        continue

    for j in range(group_num): # read 30 frames from each video , that is about 10 second
        # read 5 frames from it
        # try:
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        ret3, frame3 = cap.read()
        ret4, frame4 = cap.read()
        ret5, frame5 = cap.read()
        ret6, frame6 = cap.read()
        ret7, frame7 = cap.read()

        # except:
        #     print("\n Read path_pre fail ")
        # else:
        #     # successful write
        #     break

        if ret1 == False or ret2 == False or ret3 == False or ret4 == False or ret5 == False or ret6 == False or ret7 == False:
            break

        # frame1 = cv2.resize(frame1, tuple(dsize))
        # frame2 = cv2.resize(frame2, tuple(dsize))
        # frame3 = cv2.resize(frame3, tuple(dsize))
        # frame4 = cv2.resize(frame4, tuple(dsize))
        # frame5 = cv2.resize(frame5, tuple(dsize))
        # frame6 = cv2.resize(frame6, tuple(dsize))
        # frame7 = cv2.resize(frame7, tuple(dsize))

        # plt.figure(1)
        # plt.title("Dest Frame")
        # plt.imshow(cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB))
        # plt.show()

        # calculate the optical flow to determine if it is too normal sample
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame3_gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
        frame4_gray = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)
        frame5_gray = cv2.cvtColor(frame5, cv2.COLOR_BGR2GRAY)
        frame6_gray = cv2.cvtColor(frame6, cv2.COLOR_BGR2GRAY)
        frame7_gray = cv2.cvtColor(frame7, cv2.COLOR_BGR2GRAY)





        frame1_gray = cv2.resize(frame1_gray, tuple(resize_input))
        frame2_gray = cv2.resize(frame2_gray, tuple(resize_input))
        frame3_gray = cv2.resize(frame3_gray, tuple(resize_input))
        frame4_gray = cv2.resize(frame4_gray, tuple(resize_input))
        frame5_gray = cv2.resize(frame5_gray, tuple(resize_input))
        frame6_gray = cv2.resize(frame6_gray, tuple(resize_input))
        frame7_gray = cv2.resize(frame7_gray, tuple(resize_input))





        # cv2.imshow("frame1_gray",frame1_gray)
        # cv2.waitKey()
        # cv2.destroyAllWindows()



        flow_12 = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_23 = cv2.calcOpticalFlowFarneback(frame2_gray, frame3_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_34 = cv2.calcOpticalFlowFarneback(frame3_gray, frame4_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_45 = cv2.calcOpticalFlowFarneback(frame4_gray, frame5_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_56 = cv2.calcOpticalFlowFarneback(frame5_gray, frame6_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_67 = cv2.calcOpticalFlowFarneback(frame6_gray, frame7_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # abstract and using int32
        flow_12 = numpy.sqrt(numpy.power(flow_12[:,:,0],2) + numpy.power(flow_12[:,:,1],2))
        flow_23 = numpy.sqrt(numpy.power(flow_23[:,:,0],2) + numpy.power(flow_23[:,:,1],2))
        flow_34 = numpy.sqrt(numpy.power(flow_34[:,:,0],2) + numpy.power(flow_34[:,:,1],2))
        flow_45 = numpy.sqrt(numpy.power(flow_45[:,:,0],2) + numpy.power(flow_45[:,:,1],2))
        flow_56 = numpy.sqrt(numpy.power(flow_56[:,:,0],2) + numpy.power(flow_56[:,:,1],2))
        flow_67 = numpy.sqrt(numpy.power(flow_67[:,:,0],2) + numpy.power(flow_67[:,:,1],2))

        # we sample 50 random pixel location from the image space
        # x_list = random.sample(range(1+sample_size[0], dsize[0]-sample_size[0] ), group_sample)
        # y_list = random.sample(range(1+sample_size[1], dsize[1]-sample_size[1] ), group_sample)
        x_list = [0]
        y_list = [0]
        for sample_x, sample_y in zip(x_list, y_list):
            sample_1 = frame1[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            sample_2 = frame2[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            sample_3 = frame3[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            sample_4 = frame4[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            sample_5 = frame5[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            sample_6 = frame6[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            sample_7 = frame7[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]

            # sample_flow_12 = flow_12[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            # sample_flow_23 = flow_23[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            # sample_flow_34 = flow_34[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            # sample_flow_45 = flow_45[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            # sample_flow_56 = flow_56[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            # sample_flow_67 = flow_67[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]

            sample_flow_12 = flow_12#[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            sample_flow_23 = flow_23#[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            sample_flow_34 = flow_34#[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            sample_flow_45 = flow_45#[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            sample_flow_56 = flow_56#[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            sample_flow_67 = flow_67#[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]

            motion12 = numpy.mean(sample_flow_12)
            motion23 = numpy.mean(sample_flow_23)
            motion34 = numpy.mean(sample_flow_34)
            motion45 = numpy.mean(sample_flow_45)
            motion56 = numpy.mean(sample_flow_56)
            motion67 = numpy.mean(sample_flow_67)


            # drop = random.uniform(0,1)
            avg_motion = int((motion12 + motion23 + motion34+ motion45 + motion56 + motion67)/6.0/0.5)



            # maxmotion12 = numpy.max(sample_flow_12)
            # minmotion12 = numpy.min(sample_flow_12)

            # to skip the low motion patches
            if  motion12 < THRESHOLD_MOTION or  motion12 >= THRESHOLD_MOTION_HIGH or\
                motion23 < THRESHOLD_MOTION or  motion23 >= THRESHOLD_MOTION_HIGH or \
                motion34 < THRESHOLD_MOTION or  motion34 >= THRESHOLD_MOTION_HIGH or\
                motion45 < THRESHOLD_MOTION or  motion45 >= THRESHOLD_MOTION_HIGH or \
                motion56 < THRESHOLD_MOTION or motion56 >= THRESHOLD_MOTION_HIGH or \
                motion67 < THRESHOLD_MOTION or motion67 >= THRESHOLD_MOTION_HIGH :
                # print("\t slow motion ", motion12, motion23, motion34, motion45)
                # print("\t")
                continue

            avg_motion = avg_motion - 1# for indexing
            counter[avg_motion] %= COUNTMAX[avg_motion]
            counter[avg_motion] +=1
            # elif drop > threshold[avg_motion]:
            if not counter[avg_motion] == 1:
                # print("\t Drop it by random")
                continue


            texture_1 = frame1_gray[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            texture_2 = frame2_gray[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            texture_3 = frame3_gray[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            texture_4 = frame4_gray[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            texture_5 = frame5_gray[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            texture_6 = frame6_gray[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]
            texture_7 = frame7_gray[sample_y:sample_y + sample_size[1], sample_x: sample_x + sample_size[0]]

            # plt.figure(2)
            # plt.title("Texture Complexity")
            # plt.imshow(numpy.concatenate((texture_1, texture_2, texture_3,texture_4,texture_5),axis=0), cmap='gray')
            # plt.show()
            # cv2.imshow("texture patch 1", texture_1)
            # cv2.imshow("texture patch 2", texture_2)
            # cv2.imshow("texture patch 3", texture_3)
            # cv2.imshow("texture patch 4", texture_4)
            # cv2.imshow("texture patch 5", texture_5)
            # cv2.waitKey(0)
            # texture_1 = scipy.stats.entropy(texture_1.reshape(-1))
            ### this calculation is incorrect
            # texture_var_1 = numpy.sqrt(numpy.var(texture_1))
            # texture_var_2 = numpy.sqrt(numpy.var(texture_2))
            # texture_var_3 = numpy.sqrt(numpy.var(texture_3))
            # texture_var_4 = numpy.sqrt(numpy.var(texture_4))
            # texture_var_5 = numpy.sqrt(numpy.var(texture_5))
            #
            #
            # if texture_var_1 < THRESHOLD_TEXTURE or \
            #     texture_var_2 < THRESHOLD_TEXTURE or \
            #     texture_var_3 < THRESHOLD_TEXTURE or \
            #     texture_var_4 < THRESHOLD_TEXTURE or \
            #     texture_var_5 < THRESHOLD_TEXTURE :
            #     # print("\t low texture ",texture_var_1, texture_var_2, texture_var_3, texture_var_4, texture_var_5)
            #     continue

            dif = numpy.mean(numpy.abs((texture_1/ 255.) - (texture_7/ 255.)))
            if dif < THRESHOLD_TEXTURE_DIFF:
                print('low diff', dif)
                continue

            # to skip the low texture patches
            # sample_1 = cv2.cvtColor(sample_1, cv2.COLOR_BGR2RGB)
            # cv2.imshow("cropped 1", sample_1)
            # cv2.imshow("cropped 5", sample_5)
            # cv2.waitKey(0)

            # save as png file

            fn_new = os.path.join(OutputDir, file[:-4], "Id"+str(j)+ "_x"+ str(sample_x)+ "_y"+ str(sample_y)+  "_1.png")
            # try:
            imsave(fn_new, cv2.cvtColor(sample_1, cv2.COLOR_BGR2RGB))
            # except:
            #     print("\n Read path_pre fail ")
            # else:
            #     # successful write
            #     break

            fn_new = os.path.join(OutputDir,file[:-4] ,"Id"+str(j)+ "_x"+ str(sample_x)+ "_y"+ str(sample_y)+ "_2.png")
            # try:
            imsave(fn_new, cv2.cvtColor(sample_2, cv2.COLOR_BGR2RGB))
            # except:
            #     print("\n Read path_pre fail ")
            # else:
            #     # successful write
            #     break

            fn_new = os.path.join(OutputDir, file[:-4] ,"Id"+str(j)+ "_x"+ str(sample_x)+ "_y"+ str(sample_y)+ "_3.png")
            # try:
            imsave(fn_new, cv2.cvtColor(sample_3, cv2.COLOR_BGR2RGB))
            # except:
            #     print("\n Read path_pre fail ")
            # else:
            #     # successful write
            #     break
            fn_new = os.path.join(OutputDir, file[:-4] ,"Id"+str(j)+ "_x"+ str(sample_x)+ "_y"+ str(sample_y)+ "_4.png")
            # try:
            imsave(fn_new, cv2.cvtColor(sample_4, cv2.COLOR_BGR2RGB))
            # except:
            #     print("\n Read path_pre fail ")
            # else:
            #     # successful write
            #     break

            fn_new = os.path.join(OutputDir, file[:-4] ,"Id"+str(j)+ "_x"+ str(sample_x)+ "_y"+ str(sample_y)+ "_5.png")
            # try:
            imsave(fn_new, cv2.cvtColor(sample_5, cv2.COLOR_BGR2RGB))
            # except:
            #     print("\n Read path_pre fail ")
            # else:
            #     # successful write
            #     break
            fn_new = os.path.join(OutputDir, file[:-4] ,"Id"+str(j)+ "_x"+ str(sample_x)+ "_y"+ str(sample_y)+ "_6.png")
            # try:
            imsave(fn_new, cv2.cvtColor(sample_6, cv2.COLOR_BGR2RGB))
            # except:
            #     print("\n Read path_pre fail ")
            # else:
            #     # successful write
            #     break
            fn_new = os.path.join(OutputDir, file[:-4] ,"Id"+str(j)+ "_x"+ str(sample_x)+ "_y"+ str(sample_y)+ "_7.png")
            # try:
            imsave(fn_new, cv2.cvtColor(sample_7, cv2.COLOR_BGR2RGB))
            # except:
            #     print("\n Read path_pre fail ")
            # else:
            #     # successful write
            #     break
            im_list_rgb.append(fn_new[len(OutputDir):])

            # statistical of the motion
            motionHist[avg_motion, 0] += 1

            sample_count = sample_count + 1
            # sample_1 = sample_1.tobytes()
            # sample_2 = sample_2.tobytes()
            # sample_3 = sample_3.tobytes()
            # sample_4 = sample_4.tobytes()
            # sample_5 = sample_5.tobytes()
            # sample_6 = sample_6.tobytes()
            # sample_7 = sample_7.tobytes()
            #
            # example = tf.train.Example(features=tf.train.Features(
            #         feature={'frame1': _bytes_feature(sample_1),
            #                    'frame2': _bytes_feature(sample_2),
            #                    'frame3': _bytes_feature(sample_3),
            #                    'frame4': _bytes_feature(sample_4),
            #                    'frame5': _bytes_feature(sample_5)
            #                    }))
            # sample_count = sample_count + 1
            # print("sample count: %d" % sample_count)
            #
            # if sample_count % 20 == 18:
            #     tf_validate_writer.write(example.SerializeToString())
            # elif  sample_count % 20 == 19:
            #     tf_test_writer.write(example.SerializeToString())
            # else:
            #     tf_training_writer.write(example.SerializeToString())

    cap.release()

    if sample_count > SAMPLE_COUNT:
        print("We have %d samples, Enough"%sample_count)
        break
    else:
        # numpy.savetxt(os.path.join(OutputDir, "motionHistogram.txt"), motionHist, fmt='%.2f', delimiter=",")
        numpy.savetxt("motionHistogram.txt", motionHist, fmt='%.2f', delimiter=",")
        fl = open(os.path.join(OutputDir, "im_list_rgb.txt"), 'w')
        sep = '\n'
        fl.write(sep.join(im_list_rgb))
        fl.close()

        videoid = videoid+1
        print("Total %d videos have been scanned."%videoid)
        print("Video id. %d, Sample id %d, sample/video = %f "%(videoid,sample_count,sample_count/videoid))
# write to the file for later batch generator of KITTI set.


#write the statistc of the motion
# mhf = open(os.path.join(OutputDir, "motionHistogram.txt"), 'w')
# sep = '\n'
# mhf.write(motion)
# mhf.write(motionHist)
# mhf.close()
numpy.savetxt(os.path.join(OutputDir, "motionHistogram.txt"), motionHist,fmt='%.2f', delimiter=",")

# while(1):
#     try:
fl = open(os.path.join(OutputDir, "im_list_rgb.txt"), 'w')
sep = '\n'
fl.write(sep.join(im_list_rgb))
fl.close()
exit()
    # except:
    #     print("\n Read path_pre fail ")
    # else:
    #     break

print("done")

#
# tf_training_writer.close()
# tf_test_writer.close()
# tf_validate_writer.close()
