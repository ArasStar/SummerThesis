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
    device = torch.device('cuda:0')
else:
    print("CUDA didn't work")
    device = torch.device('cpu')


#print(os.getcwd())
root_dir = '/../../'
#root_dir = '/../'

show=0
learning_rate=0.0001
batch_size=16


method="relative_position"
#Params for relative_position
split = 3.0

#Params for jigsaw
grid_crop_size=225
patch_crop_size=64
perm_set_size=300
num_epochs=3

def self_train(method="relative_position",num_epochs=3, learning_rate=0.0001, batch_size=16,split = 3.0,
                                                                                            grid_crop_size=225,patch_crop_size=64,perm_set_size=300):

  #just ToTensor before pathch
  transform_train= transforms.Compose([  transforms.RandomCrop(320), transforms.RandomVerticalFlip(), transforms.ToTensor()])

  #after patch transformation
  transform_after_patching= transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])                                 

  labels_path="/home/aras/Desktop/SummerThesis/code/custom_lib/chexpert_load/self_train_labels.pt"
  cheXpert_train_dataset, dataloader = chexpert_load.chexpert_load("/home/aras/Desktop/SummerThesis/code/custom_lib/chexpert_load/self_train.csv",
                                                                  transform_train,batch_size, labels_path=labels_path,root_dir = root_dir)

  if method == "jigsaw":
    file_name_p_set = F"permutation_set{perm_set_size}.pt"
    PATH_p_set = F"/home/aras/Desktop/SummerThesis/code/custom_lib/permutation_set/saved_permutation_sets/{file_name_p_set}"
    model = models.densenet121(num_classes = perm_set_size)
    model.classifier = jigsaw.Basic_JigsawHead(1024,perm_set_size, gpu = use_cuda)
    
  elif method =="relative_position":
    model = models.densenet121(num_classes = 8)
    model.classifier = patch.Basic_RelativePositionHead(1024, gpu = use_cuda)

  model=model.to(device=device)
  
  
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  criterion = torch.nn.CrossEntropyLoss().to(device = device)

  currentDT = datetime.datetime.now()

  if method == "jigsaw":
    print(F"{method}_perm_set_size{perm_set_size}_epoch{num_epochs}_batch{batch_size}_grid_size{grid_crop_size}_patch_size{patch_crop_size}_CUDA{use_cuda}")
  elif method == "relative_position":
    print(F"{method}_epoch_{num_epochs}_batch{batch_size}_split{split}_CUDA{use_cuda}")

  plot_loss = []
  for epoch in range(num_epochs):
      for i,  (images, observations) in enumerate(dataloader):   # Load a batch of images with its (index, data, class)
        
        if method == "jigsaw":
          patcher = jigsaw.Jigsaw(images, PATH_p_set, grid_crop_size=grid_crop_size, patch_crop_size=patch_crop_size,
                                        transform =transform_after_patching, gpu = use_cuda, show=show)  
        elif method=="relative_position":
          patcher = patch.Patch(images,split=split,transform=transform_after_patching,show=show)

        patches, labels =  patcher()
        patches = patches.to(device=device, dtype=torch.float32)
        
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
        

        if i%10 == 0:
                plot_loss.append(loss)
                
        if (i+1) % 50 == 0:                              # Logging
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                    %(epoch+1, num_epochs, i+1, len(cheXpert_train_dataset)//batch_size, loss))


  print('training done')


  aftertDT = datetime.datetime.now()
  c=aftertDT-currentDT
  mins,sec=divmod(c.days * 86400 + c.seconds, 60)

  print(mins,"mins ", sec,"secs")

  if method == "jigsaw":
    print(F"{method}perm_set_size{perm_set_size}_epoch{num_epochs}_batch{batch_size}_grid_size{grid_crop_size}_patch_size{patch_crop_size}_CUDA{use_cuda}")
    file_name = F"{method}perm_set_size{perm_set_size}_epoch{epoch}_batch{batch_size}_grid_size{grid_crop_size}_patch_size{patch_crop_size}.tar"

  elif method == "relative_position":
      file_name = F"{method}_epoch{epoch}_batch{batch_size}_split{split}.tar"
      print(F"{method}_epoch{num_epochs}_batch{batch_size}_split{split}_CUDA{use_cuda}")

  

  PATH = F"/home/aras/Desktop/saved_models/{file_name}"

  torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss':plot_loss}, PATH)

  #torch.save(model.state_dict(), PATH)
  print('saved  model(model,optim,loss, epoch)')# to google drive')

  '''
  fig=plt.figure()
  fig.suptitle("loss plot")
  plt.plot(plot_loss)
  plt.xlabel('iterations')
  plt.ylabel('loss')
  plt.show()
  '''




#'''(method="relative_position",num_epochs=3, learning_rate=0.0001, batch_size=16,split = 3.0, grid_crop_size=225,patch_crop_size=64,perm_set_size=300)'''

show=0
learning_rate=0.0001
batch_size=16
resize=320


method="relative_position"
#Params for relative_position
split = 3.0
#Params for jigsaw
grid_crop_size=225
patch_crop_size=64
perm_set_size=300
num_epochs=3


schedule=[{"method":"relative_position","num_epochs":3},
          {"method":"jigsaw","num_epochs":3},
          {"method":"relative_position","num_epochs":3,"split":2}]


for kwargs in schedule:
  self_train(**kwargs)


