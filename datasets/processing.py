# -*- coding: utf-8 -*-

import os
import torch

from glob import glob
import numpy as np
import torch.utils.data as Data

import torchvision.transforms.functional as TF

from PIL import Image
import numpy as np
from skimage.io import imread, imsave

from glob import glob
import math
import cv2
import os
from PIL import Image
import time


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
            #print(imagelist)
        return imagelist


def new_folder(result_root):
    if not os.path.exists(result_root):
        os.makedirs(result_root)

def processing_testsets_synthetic(scale,data_path,save_path,dataset_name):
    Depth_files = sorted(get_img_file(os.path.join(data_path, 'Depth')))
    #RGB_files = sorted(get_img_file(os.path.join(data_path, 'RGB')))
    #assert len(Depth_files) == len(RGB_files)
    
    for image_index in range(len(Depth_files)):
        print(Depth_files[image_index])
        #image = np.transpose(imread(RGB_files[image_index]).astype(
        #    'float32'), [2, 0, 1]) # [3,h,w] 0~255
        if dataset_name == 'Middlebury' or dataset_name == 'Lu':
            depth_hr = imread(Depth_files[image_index]).astype(
                'float32')   # [h,w] 0~255
        elif dataset_name == 'NYU':
            depth_hr = np.load(Depth_files[image_index]).astype(
                'float32')
        elif dataset_name == 'RGBDD':
            depth_hr = imread(Depth_files[image_index]).astype(
                'float32') /1000  # [h,w] 0~3000+(mm)
        h, w = depth_hr.shape
        #print(depth_hr.shape)
    
        # crop to make divisible
        h = h - h % scale
        w = w - w % scale
        #image = image[:, :h, :w]
        depth_hr = 1 / depth_hr[:h, :w]
    
        # get LR Depth map
        """ h, w = depth_hr.shape
        depth_lr = np.array(Image.fromarray(depth_hr).resize(
            (w//scale, h//scale), Image.BICUBIC))  # bicubic """
    
        # normalize depth map
        depth_min = depth_hr.min()
        depth_max = depth_hr.max()
        print(depth_min, depth_max)
        if depth_min == depth_max: continue
        assert depth_min != depth_max
        # depth_hr_norm = (depth_hr - depth_min) / (depth_max - depth_min)
        """ depth_lr_norm = (depth_lr - depth_min) / (depth_max - depth_min)
        print(depth_lr_norm.min(), depth_lr_norm.max())
     """
        depth_hr_norm = (depth_hr - depth_min) / (depth_max - depth_min)
        depth_lr_norm = np.array(Image.fromarray(depth_hr_norm).resize(
            (w//scale, h//scale), Image.BICUBIC
        ))
        # normalize RGB image
        #image = image.astype(np.float32)/255  # [3, H, W] 0~1
        #image_norm = (image - np.array([0.485, 0.456, 0.406]).reshape(3,
        #         1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
        # follow DKN & FDSR use bicubic upsampling of PIL
        #depth_lr_up = np.array(Image.fromarray(
        #    depth_lr_norm).resize((w, h), Image.BICUBIC))
    
        new_folder(os.path.join(save_path, 'DepthHR'))
        #new_folder(os.path.join(save_path, 'RGB'))
        new_folder(os.path.join(save_path, 'DepthLr'))
    
        # save the patches in npy
        filename = os.path.splitext(Depth_files[image_index].split('/')[-1])[0]
        np.save(os.path.join(save_path, 'DepthHR',
                filename+'-depthHr.npy'), depth_hr[None, :, :].astype(np.float32))
        #np.save(os.path.join(save_path, 'RGB',
        #        filename+'-RGB.npy'), image_norm.astype(np.float32))
        np.save(os.path.join(save_path, 'DepthLr',
                filename+'-depthLr.npy'), depth_lr_norm[None, :, :].astype(np.float32))

processing_testsets_synthetic(4,'/data/weimin_data/nyu','/data/weimin_data/nyu/x4_t', 'NYU')
