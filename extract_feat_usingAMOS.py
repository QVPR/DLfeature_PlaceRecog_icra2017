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

layerNames = ['conv3', 'conv4', 'conv5', 'conv6' ,'pool1', 'pool2', 'fc7_new', 'fc8_new']
parser = argparse.ArgumentParser(description='AmosNet-HybridNet')
parser.add_argument('--model', '-m', type=str, default='HybridNet', choices=['HybridNet','AmosNet'])
parser.add_argument('--imgDirPath', '-p', type=str, default='/work/qvpr/data/ready/gt_aligned/sample_2014-Multi-Lane-Road-Sideways-Camera/NIL/images/')
parser.add_argument('--uniqueSaveStr', '-u', type=str, default='',help='insert this unique string in the save path')
parser.add_argument('--layerName', '-l', type=str, default='fc7_new',help='layer to extract features from', choices=layerNames)

def main():
    opt = parser.parse_args()
    print(opt)
    modelName = opt.model
    modelPath = "/work/qvpr/models/{}/".format(modelName)
    datasetPath = opt.imgDirPath
    uniFileName = opt.uniqueSaveStr
    mode = 'cpu'
    layerName = opt.layerName

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
    layerDims = [64896,64896,43264,43264,69984,43264,4096,2543]
    layerDims = dict(zip(layerNames,layerDims))

    saveName = '{}_'.format(modelName) + uniFileName + '_feat_{}'.format(layerName); # Path to save extratced feature vector

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

    lst_images = sorted(os.listdir(datasetPath))#[:2000]
    lst_images = [f for f in lst_images if '.png' in f or '.jpg' in f]

    i = 0;
    tt = len(lst_images);
    print("Number of images to process: ", tt)
    feats = []
    print("Extracting features from layer {} with dims {} ".format(layerName,layerDims[layerName]))
    for img in lst_images:
        imData = caffe.io.load_image(os.path.join(datasetPath,img))#[:800,:,:]
        net.blobs['data'].data[...] = transformer.preprocess('data', imData)
        out = net.forward()

        fea = np.squeeze(net.blobs[layerName].data);
        feats.append(fea.flatten())
        print(i);
        i += 1;

    #scipy.io.savemat(cv6_save,{'fea_cv6':fea_cv6});
    # scipy.io.savemat(fc7_save,{'fea_fc7':fea_fc7});
    #scipy.io.savemat(fc8_save,{'fea_fc8':fea_fc8});
    print("Saved features at: ", saveName)
    np.save(saveName,feats)


if __name__ == "__main__":
    main()