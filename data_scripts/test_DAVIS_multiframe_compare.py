
import itertools
import time
import os
import torch
from torch.autograd import Variable
from shutil import copyfile
import numpy
import random
import numpy as np
from scipy.misc import imsave, imread
import networks
from my_args import  args
from AverageMeter import  *
torch.backends.cudnn.benchmark = True # to speed up the
MB_EVAL_DATA = "/tmp4/wenbobao_data/HD/DAVIS"
MB_EVAL_RESULT = "/tmp4/wenbobao_data/HD/DAVIS_ours"
if not os.path.exists(MB_EVAL_RESULT):
    os.mkdir(MB_EVAL_RESULT)

model = networks.__dict__[args.netName](channel=args.channels,
                                    filter_size = args.filter_size ,
                                    timestep=args.time_step,
                                    training=False)
if args.use_cuda:
    model = model.cuda()

args.SAVED_MODEL = './model_weights/best.pth'
if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
        # model.load_state_dict(torch.load(args.SAVED_MODEL))

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We donn't load any trained weights")
    print("*****************************************************************")

model = model.eval() # deploy mode



timestep = args.time_step
numFrames = int(1.0 / timestep) - 1
time_offsets = [kk * timestep for kk in range(1, 1 + numFrames, 1)]


use_cuda=args.use_cuda
save_which=args.save_which
dtype = args.dtype
unique_id =str(random.randint(0, 100000))
print("The unique id for current testing is: " + str(unique_id))

subdir = sorted(os.listdir(MB_EVAL_DATA))

gen_dir = os.path.join(MB_EVAL_RESULT, unique_id)
gen_dir0 = os.path.join(MB_EVAL_RESULT,unique_id+str(0))
os.mkdir(gen_dir)
os.mkdir(gen_dir0)
for dir in subdir:
    print(dir)
    if not dir == "train" and not dir == "blackswan":
        continue
    # prepare the image save path
    os.mkdir(os.path.join(gen_dir, dir))
    os.mkdir(os.path.join(gen_dir0, dir))

    files = sorted(os.listdir(os.path.join(MB_EVAL_DATA, dir)))

    # the last frame has no following frame
    for step in range(0, len(files)-1):
        file1 = files[step]
        file2 = files[step+1]
        print(file1, '\n', file2)

        arguments_strFirst = os.path.join(MB_EVAL_DATA, dir, file1)
        arguments_strSecond = os.path.join(MB_EVAL_DATA, dir, file2)
                
        X0_all =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype)
        X1_all =  torch.from_numpy( np.transpose(imread(arguments_strSecond), (2,0,1)).astype("float32")/ 255.0).type(dtype)
        intWidth_ori = X0_all.size(2)
        intHeight_ori = X0_all.size(1)
        split_lengthY = 256 # 512 #intHeight_ori/split_num
        split_lengthX = 448 #896 #intWidth_ori/ split_num
        intPaddingRight = int(float(intWidth_ori)/split_lengthX + 1) * split_lengthX  - intWidth_ori
        intPaddingBottom = int(float(intHeight_ori)/split_lengthY + 1) * split_lengthY - intHeight_ori
        # intPaddingRight = 0 if intPaddingRight == split_lengthX else intPaddingRight
        # intPaddingBottom = 0 if intPaddingBottom == split_lengthY else intPaddingBottom
        pader0 = torch.nn.ReplicationPad2d([0, intPaddingRight , 0, intPaddingBottom])
        print("Init pad right/bottom " + str(intPaddingRight) + " / " + str(intPaddingBottom))
            
        intPaddingRight = 32 #64# 128# 256
        intPaddingLeft = 32 #64 #128# 256
        intPaddingTop = 32 #64 #128#256
        intPaddingBottom = 32#64 # 128# 256
        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

        torch.set_grad_enabled(False)
        X0_all = torch.unsqueeze(X0_all,0)
        X1_all = torch.unsqueeze(X1_all,0)
        X0_all =  Variable(X0_all)
        X1_all =  Variable(X1_all)

  

        X0_all =  pader0(X0_all)
        X1_all =  pader0(X1_all)
        y_all = [np.zeros((X0_all.size(2), X1_all.size(3) ,3 ), dtype = "float32") for _ in range(0, numFrames)]
        y_all0 =[ np.zeros((X0_all.size(2), X1_all.size(3), 3), dtype="float32") for _ in range(0,numFrames)]

        X0_all =  pader(X0_all)
        X1_all =  pader(X1_all)
                
        assert (split_lengthY == int(split_lengthY) and split_lengthX  == int(split_lengthX))
        split_lengthY = int(split_lengthY)
        split_lengthX = int(split_lengthX)
        split_numY = int(float(intHeight_ori)/split_lengthY + 1)
        split_numX = int(float(intWidth_ori)/split_lengthX + 1)
        splitsY = range(0,split_numY)
        splitsX = range(0,split_numX)

        intWidth = split_lengthX
        intWidth_pad  = intWidth + intPaddingLeft + intPaddingRight
        intHeight = split_lengthY
        intHeight_pad = intHeight + intPaddingTop + intPaddingBottom
                
        print("split " + str(split_numY) + ' , ' + str(split_numX))
        for split_j, split_i in itertools.product(splitsY,splitsX):
            print(str(split_j) + ", \t " + str(split_i))
            X0 = X0_all[:,:,split_j * split_lengthY :(split_j + 1)*split_lengthY + intPaddingBottom + intPaddingTop ,
                split_i * split_lengthX :(split_i+ 1) * split_lengthX + intPaddingRight + intPaddingLeft]
            X1 = X1_all[:,:,split_j * split_lengthY:(split_j + 1)*split_lengthY  + intPaddingBottom + intPaddingTop ,
                            split_i * split_lengthX:(split_i+ 1) * split_lengthX + intPaddingRight + intPaddingLeft]
            y_ = torch.FloatTensor()
                        
            assert (X0.size(1) == X1.size(1))
            assert (X0.size(2) == X1.size(2))

            if use_cuda:
                X0 = X0.cuda()
                X1 = X1.cuda()

            y_s ,offset,filter = model(torch.stack((X0, X1),dim = 0))
            # y_s  = [ [X0] * numFrames ] * 2
            y_ = y_s[save_which]
            y_0 = y_s[0]

            if use_cuda:
                X0 = X0.data.cpu().numpy()
                y_ = [item.data.cpu().numpy() for item in y_]
                y_0 = [item.data.cpu().numpy() for item in y_0]
                X1 = X1.data.cpu().numpy()
            else:
                X0 = X0.data.numpy()
                y_ = y_.data.numpy()
                X1 = X1.data.numpy()

            X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
            y_ = [np.transpose(255.0 * item.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight,
                                  intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]
            y_0 = [np.transpose(255.0 * item.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight,
                              intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for item in y_0]

            X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

            for kkk in range(0, len(y_)):
                y_all[kkk][split_j * split_lengthY:(split_j + 1) * split_lengthY,
                    split_i * split_lengthX:(split_i + 1) * split_lengthX, :] = \
                    np.round(y_[kkk]).astype(numpy.uint8)
                y_all0[kkk][split_j * split_lengthY:(split_j + 1) * split_lengthY,
                        split_i * split_lengthX:(split_i + 1) * split_lengthX, :] = \
                        np.round(y_0[kkk]).astype(numpy.uint8)

        arguments_strOut = os.path.join(gen_dir, dir, "{:0>6d}.jpg".format(step * (numFrames+1)+ 0))
        copyfile(arguments_strFirst,arguments_strOut)
        print(arguments_strOut)

        arguments_strOut0 = os.path.join(gen_dir0, dir, "{:0>6d}.jpg".format(step * (numFrames+1)+ 0))
        copyfile(arguments_strFirst,arguments_strOut0)
        print(arguments_strOut0)


        indexes = list(range(1, 1 + numFrames, 1))

        for item, item0,index   in zip(y_all,y_all0,indexes):
            item =item[:intHeight_ori, :intWidth_ori, :]
            item0 = item0[:intHeight_ori, :intWidth_ori, :]

            arguments_strOut = os.path.join(gen_dir, dir, "{:0>6d}.jpg".format(step *(numFrames +1) + index))

            imsave(arguments_strOut, np.round(item).astype(numpy.uint8))
            print(arguments_strOut)

            arguments_strOut0 = os.path.join(gen_dir0, dir, "{:0>6d}.jpg".format(step *(numFrames +1) + index))
            #imsave(arguments_strOut0, np.round(item0).astype(numpy.uint8))
            copyfile(arguments_strFirst,arguments_strOut0)            
            print(arguments_strOut0)
        arguments_strOut = os.path.join(gen_dir, dir, "{:0>6d}.jpg".format((step + 1) * (numFrames + 1)))
        copyfile(arguments_strSecond, arguments_strOut)

        arguments_strOut0 = os.path.join(gen_dir0, dir, "{:0>6d}.jpg".format((step + 1) * (numFrames + 1)))
        copyfile(arguments_strSecond, arguments_strOut0)

    #exit(0)