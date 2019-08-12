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


root_PATH = "/homes/ay1218/Desktop/"
path = "/vol/bitbucket/ay1218/saved_models/self_supervised/dummynaive_combination*Rotation*Relative_Position*Jigsaw*_num_epochs2_batch_size4_learning_rate0.0001_K4_resize128_patch_size96_perm_set_size1000_grid_crop_size225_patch_crop_size64/DUMMYnaive_combination*Rotation*Relative_Position*Jigsaw*_num_epochs2_batch_size4_learning_rate0.0001_K4_resize128_patch_size96_perm_set_size1000_grid_crop_size225_patch_crop_size64.tar"
val =task_validation.Task_Validation(path)
#val =task_validation.Task_Validation("/vol/bitbucket/ay1218/saved_models/self_supervised/naive_combination*Relative_Position*Jigsaw*_num_epochs2_batch_size16_learning_rate0.0001_patch_size64_perm_set_size100_grid_crop_size225_patch_crop_size64/naive_combination*Relative_Position*Jigsaw*_num_epochs2_batch_size16_learning_rate0.0001_patch_size64_perm_set_size100_grid_crop_size225_patch_crop_size64.tar")
val()


import os

directory = saved_model_PATH+"saved_models/self_supervised/"
'''
count =0
print(directory)
for filename in os.listdir(directory):
    print(filename)
    if not file_name.__contains__("eski")
        val = task_validation.Task_Validation(directory+filename+"/"+filename+".tar")
        val()
        if count>10:
            break
            count+=1


'''
print("finish")





















#hop
