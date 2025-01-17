# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import os#for CUDA tracking
#from __future__ import print_function, division
import torch
import pandas as pd
#from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#MODELS
import re
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

import datetime
import sys

import task_validation

saved_model_PATH="/vol/bitbucket/ay1218/"

root_PATH_dataset = "/vol/gpudata/ay1218/"
#root_PATH_dataset = saved_model_PATH

path=[]
x="/vol/bitbucket/ay1218/saved_models/semi_supervised/protect_self_supervised0.1_CC_GAN2*Jigsaw*Relative_Position*Rotation*num_epochs3learning_rate0.0002batch_size16resize128label_rate0.2_perm_set_size1000_grid_crop_size225_patch_crop_size64__patch_size96__K4_resize128/self_supervised0.1_CC_GAN2*Jigsaw*Relative_Position*Rotation*num_epochs3learning_rate0.0002batch_size16resize128label_rate0.2_perm_set_size1000_grid_crop_size225_patch_crop_size64__patch_size96__K4_resize128.tar"
y="/vol/bitbucket/ay1218/saved_models/semi_supervised/protect_self_supervised0.1_CC_GAN2*Rotation*num_epochs3learning_rate0.0002batch_size16resize128label_rate0.2_K4_resize128/self_supervised0.1_CC_GAN2*Rotation*num_epochs3learning_rate0.0002batch_size16resize128label_rate0.2_K4_resize128.tar"
z="/vol/bitbucket/ay1218/saved_models/semi_supervised/protect_self_supervised0.1_CC_GAN*Jigsaw*Relative_Position*Rotation*num_epochs3learning_rate0.0002batch_size16resize128label_rate0.2_perm_set_size1000_grid_crop_size225_patch_crop_size64__patch_size96__K4_resize128/self_supervised0.1_CC_GAN*Jigsaw*Relative_Position*Rotation*num_epochs3learning_rate0.0002batch_size16resize128label_rate0.2_perm_set_size1000_grid_crop_size225_patch_crop_size64__patch_size96__K4_resize128.tar"
k="/vol/bitbucket/ay1218/saved_models/semi_supervised/protect_self_supervised0.1_CC_GAN*Rotation*num_epochs3learning_rate0.0002batch_size16resize128label_rate0.2_K4_resize128/self_supervised0.1_CC_GAN*Rotation*num_epochs3learning_rate0.0002batch_size16resize128label_rate0.2_K4_resize128.tar"
#path="/vol/bitbucket/ay1218/saved_models/self_supervised/Relative_Position_num_epochs4_batch_size16_learning_rate1e-06_patch_size96/Relative_Position_num_epochs4_batch_size16_learning_rate1e-06_patch_size96.tar"
path.append(x)
path.append(y)
path.append(z)
path.append(k)

for p in path:
    #path = "/vol/bitbucket/ay1218/saved_models/self_supervised/naive_combination*Rotation*Relative_Position*Jigsaw*_num_epochs2_batch_size16_learning_rate0.0001_K4_resize320_patch_size96_perm_set_size1000_grid_crop_size225_patch_crop_size64/naive_combination*Rotation*Relative_Position*Jigsaw*_num_epochs2_batch_size16_learning_rate0.0001_K4_resize320_patch_size96_perm_set_size1000_grid_crop_size225_patch_crop_size64.tar"
    val =task_validation.Task_Validation(p,normalize=False,gpu= False)
    #val =task_validation.Task_Validation("/vol/bitbucket/ay1218/saved_models/self_supervised/naive_combination*Relative_Position*Jigsaw*_num_epochs2_batch_size16_learning_rate0.0001_patch_size64_perm_set_size100_grid_crop_size225_patch_crop_size64/naive_combination*Relative_Position*Jigsaw*_num_epochs2_batch_size16_learning_rate0.0001_patch_size64_perm_set_size100_grid_crop_size225_patch_crop_size64.tar")
    val()

#
# import os
#
# normalize=True
# directory = saved_model_PATH+"saved_models/self_supervised/"
#
# count =0
# print(directory)
# for filename in os.listdir(directory):
#     print(filename.split("/")[-1])
#     if not filename.__contains__("eski"):
#         val = task_validation.Task_Validation(directory+filename+"/"+filename+".tar",normalize=normalize)
#         val()
#         if count==-10:
#             break
#             count+=1
#
#

print("finish")





















#hop
