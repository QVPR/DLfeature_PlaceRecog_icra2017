#!/usr/bin/env python

# how to use: need to 

# Purpose: Generate features from a file list
# Author:  Zetao Chen (PhD candidate of Cyphy Lab)
# Date:    17/06/15. Modifed: 18/08/2018

############################Library########################################
# Make sure that caffe is on the python path:
# file_root = '/home/jason/Install/caffe-master/'  # point this to the root directory of your caffe packages
# import sys
# sys.path.insert(0, file_root + 'python')
import caffe

import numpy as np;
import scipy.io
import string
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='AmosNet-HybridNet')
parser.add_argument('--model', '-m', type=str, default='HybridNet', choices=['HybridNet','AmosNet'])
parser.add_argument('--imgDirPath', '-p', type=str, default='/work/qvpr/data/ready/gt_aligned/sample_2014-Multi-Lane-Road-Sideways-Camera/NIL/images/')
parser.add_argument('--uniqueSaveStr', '-u', type=str, default='',help='insert this unique string in the save path')

def main():
    opt = parser.parse_args()
    print(opt)
    modelName = opt.model
    modelPath = "/work/qvpr/models/{}/".format(modelName)
    datasetPath = opt.imgDirPath
    uniFileName = opt.uniqueSaveStr
    mode = 'cpu'

    if mode == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    elif mode == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
    ############################################### PATH ########################################
    # use "print [(k, v.data.shape) for k, v in net.blobs.items()]" to check the dimensions of the corresponding layers.
    pool1_dim = 69984; # 96*27*27
    pool2_dim = 43264;# 256*13*13
    cv3_dim = 64896; # 384*13*13
    cv4_dim = 64896; # 384*13*13
    cv5_dim = 43264; # 256*13*13
    cv6_dim = 43264; # 256*169
    fc7_dim = 4096; # Feature Dimension
    fc8_dim = 2543; # keep these features dim fixed as they need to match the network architecture inside "HybridNet"
    folder = ''
    fileidx = ''

    # images_file =  dataset + '.txt'; # This is my image list [img_path...], each line specifies the location of one image file

    # uncomment the line you want to extract features from layers other than fc7. 
    #cv1_save = 'conv1.mat'; # Path to save extratced feature vector
    #cv2_save = 'conv2.mat'; # Path to save extratced feature vector
    #cv3_save = 'conv3.mat'; # Path to save extratced feature vector
    #cv4_save = 'conv4.mat'; # Path to save extratced feature vector
    #cv5_save = 'conv5.mat'; # Path to save extratced feature vector
    #cv6_save = dataset + '_feat/conv6.mat'; # Path to save extratced feature vector
    fc7_save = '{}_'.format(modelName) + uniFileName + '_feat_fc7'; # Path to save extratced feature vector
    #fc8_save = dataset + '_feat/fc8.mat'; # Path to save extratced feature vector

    model_file  = os.path.join(modelPath,'{}.caffemodel'.format(modelName)); # This is my pre-trained caffe model
    #print model_file
    ####################################setup caffe########################################

    if mode == 'cpu':
        caffe.set_mode_cpu()
    elif mode == 'gpu':
        caffe.set_mode_gpu()

    net = caffe.Net(os.path.join(modelPath,'deploy.prototxt'),
                    model_file,
                    caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(os.path.join(modelPath,'amosnet_mean.npy')).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    net.blobs['data'].reshape(1,3,227,227)

    # file_images   = open(images_file);
    # buffer_images = file_images.readlines();

    # lst_images = [];
    # for index in range(0,len(buffer_images)):
    #     lst_images.append(string.replace(buffer_images[index].split()[0],'\n','')); #eliminate \n

    lst_images = sorted(os.listdir(datasetPath))#[:2000]
    lst_images = [f for f in lst_images if '.png' in f or '.jpg' in f]

    #fea_cv1 = np.zeros((len(lst_images),pool1_dim));
    #fea_cv1 = np.zeros((len(lst_images),96,27,27));
    #fea_cv2 = np.zeros((len(lst_images),256,13,13));
    #fea_cv3 = np.zeros((len(lst_images),384,13,13));
    #fea_cv4 = np.zeros((len(lst_images),384,13,13));
    #fea_cv5 = np.zeros((len(lst_images),256,13,13));
    #fea_cv6 = np.zeros((len(lst_images),256,13,13));
    fea_fc7 = np.zeros((len(lst_images),fc7_dim));
    #fea_fc8 = np.zeros((len(lst_images),fc8_dim));

    i = 0;
    tt = len(lst_images);
    print(tt);
    for img in lst_images:
        imData = caffe.io.load_image(os.path.join(datasetPath,img))#[:800,:,:]
        net.blobs['data'].data[...] = transformer.preprocess('data', imData)
        out = net.forward()

      #  fea = np.squeeze(net.blobs['pool1'].data); # Extract feature from fc layer
       # fea_cv1[i,:] = fea;

        #fea = np.squeeze(net.blobs['pool2'].data); # Extract feature from fc layer
        #fea_cv2[i,:] = fea;

        #fea = np.squeeze(net.blobs['conv3'].data); # Extract feature from fc layer
        #fea_cv3[i,:] = fea;

        #fea = np.squeeze(net.blobs['conv4'].data); # Extract feature from fc layer
        #fea_cv4[i,:] = fea;

        #fea = np.squeeze(net.blobs['conv5'].data); # Extract feature from fc layer
        #fea_cv5[i,:] = fea;

        #fea = np.squeeze(net.blobs['conv6'].data); # Extract feature from fc layer
        #fea_cv6[i,:] = fea;

        fea = np.squeeze(net.blobs['fc7_new'].data);
        fea_fc7[i,:] = fea;

        #fea = np.squeeze(net.blobs['fc8_new'].data);
        #fea_fc8[i,:] = fea;

        print(i);
        i += 1;

    #scipy.io.savemat(cv6_save,{'fea_cv6':fea_cv6});
    # scipy.io.savemat(fc7_save,{'fea_fc7':fea_fc7});
    #scipy.io.savemat(fc8_save,{'fea_fc8':fea_fc8});
    np.save(fc7_save,fea_fc7)


if __name__ == "__main__":
    main()