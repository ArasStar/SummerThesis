# -*- coding: utf-8 -*-
"""self_supervised.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gELTxi1EabmwWVGviJWam-O19w4Arei7
"""
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import os#for CUDA tracking
#from __future__ import print_function, division
import torch
import pandas as pd
from skimage.io import imread
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

sys.path.insert(0, '/home/aras/Desktop/SummerThesis/code/custom_lib/chexpert_load')
sys.path.insert(0, '/home/aras/Desktop/SummerThesis/code/custom_lib/patch')
sys.path.insert(0, '/home/aras/Desktop/SummerThesis/code/custom_lib/jigsaw')

'''
#for colab
#Patch
if "gdrive/My Drive/summerthesis/custom_lib/patch" not in sys.path:
  sys.path.append("gdrive/My Drive/summerthesis/custom_lib/patch")

if "gdrive/My Drive/summerthesis/custom_lib/chexpert_load" not in sys.path:
  sys.path.append("gdrive/My Drive/summerthesis/custom_lib/chexpert_load")
  
  #Patch
if "gdrive/My Drive/summerthesis/custom_lib/jigsaw" not in sys.path:
  sys.path.append("gdrive/My Drive/summerthesis/custom_lib/jigsaw")
'''

import chexpert_load
import patch
import jigsaw

use_cuda = True
if use_cuda and torch.cuda.is_available():
    print("using CUDA")
    device = torch.device('cuda')
else:
    print("CUDA didn't work")
    device = torch.device('cpu')


root_dir = '/../../'
learning_rate=0.0001
batch_size=16
dtype = torch.float32
resize=320


method="jigsaw"
show=True
#Params for relative_position
split = 3.0

#Params for jigsaw
grid_crop_size=225
patch_crop_size=64
num_classes=300

file_name_p_set = F"permutation_set{num_classes}.pt"
PATH_p_set = F"/home/aras/Desktop/SummerThesis/code/custom_lib/permutation_set/saved_permutation_sets/{file_name_p_set}"

#just ToTensor before pathch
transform_train= transforms.Compose([          transforms.RandomCrop(320),transforms.ToTensor()])

#after patch transformation
transform_after_patching= transforms.Compose([ transforms.ToPILImage(),transforms.ToTensor(),
                                               transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])                                 

labels_path="/home/aras/Desktop/SummerThesis/code/custom_lib/chexpert_load/labels.pt"
cheXpert_train_dataset, dataloader = chexpert_load.chexpert_load("/home/aras/Desktop/CheXpert-v1.0-small/train.csv",
                                                                 transform_train,batch_size,labels_path, root_dir = root_dir)

model = models.densenet121(num_classes = num_classes)

if method == "jigsaw":
  model.classifier = jigsaw.Basic_JigsawHead(1024,num_classes, gpu = use_cuda)
  
elif method=="relative_position":
  model.classifier = patch.Basic_RelativePositionHead(1024, gpu = use_cuda)

model=model.to(device=device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion =torch.nn.CrossEntropyLoss().to(device = device)

currentDT = datetime.datetime.now()

num_epochs=1
print(F"{method}_num_classes{num_classes}_epoch{num_epochs}_batch{batch_size}_grid_size{grid_crop_size}_patch_size{patch_crop_size}_CUDA{use_cuda}")

for epoch in range(num_epochs):
    for i,  (images, observations) in enumerate(dataloader):   # Load a batch of images with its (index, data, class)
      
      if method == "jigsaw":
        patcher = jigsaw.Jigsaw(images, PATH_p_set, grid_crop_size=grid_crop_size, patch_crop_size=patch_crop_size,
                                       transform =transform_after_patching, gpu = use_cuda, show=show)  
      elif method=="relative_position":
        patcher = patch.Patch(images,split=split,transform=transform_after_patching,show=show)

      patches, labels =  patcher()
      patches = patches.to(device=device, dtype=dtype)
      
      if show:
        print("showa giriyooor",show)
        break
        
      labels = labels.to(device=device, dtype=torch.long)
        
      #break
      outputs = model(patches)              # Forward pass: compute the output class given a image
        
      loss = criterion(outputs, labels)           # Compute the loss: difference between the output class and the pre-given label

      optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
      loss.backward()                                   # Backward pass: compute the weight
      optimizer.step()                                  # Optimizer: update the weights of hidden nodes
        
        
      if (i+1) % 50 == 0:                              # Logging
          print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1, len(cheXpert_train_dataset)//batch_size, loss))


print('training done')


aftertDT = datetime.datetime.now()
c=aftertDT-currentDT
mins,sec=divmod(c.days * 86400 + c.seconds, 60)


print(F"{method}_num_classes{num_classes}_epoch{num_epochs}_batch{batch_size}_grid_size{grid_crop_size}_patch_size{patch_crop_size}_CUDA{use_cuda}")
print(mins,"mins ", sec,"secs")

file_name = F"jigsaw_num_classes{num_classes}_epoch{epoch}_batch{batch_size}_grid_size{grid_crop_size}_patch_size{patch_crop_size}.tar"

if not show:
  PATH=F"/content/gdrive/My Drive/summerthesis/saved_model/{file_name}"


PATH = F"/home/aras/Desktop/saved_models/{file_name}"

torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)

#torch.save(model.state_dict(), PATH)
print('saved  model(model,optim,loss, epoch)')# to google drive')