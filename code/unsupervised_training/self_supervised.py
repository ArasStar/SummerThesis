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

root_PATH = "/home/aras/Desktop/"
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/chexpert_load')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/patch')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/jigsaw')

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

def self_train(method="relative_position",num_epochs=3, learning_rate=0.0001, batch_size=16,split = 3.0,
                                                                                            grid_crop_size=225,patch_crop_size=64,perm_set_size=300 , from_checkpoint=None, root_PATH = "/home/aras/Desktop/"):
  

  model = models.densenet121()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  criterion = torch.nn.CrossEntropyLoss().to(device = device)
  plot_loss = []

  if from_checkpoint:
        method = from_checkpoint[from_checkpoint.index("saved_models/")+13:from_checkpoint.index('_epoch')]


  

  if method == "jigsaw":
    head_patch = patch.Basic_RelativePositionHead(1024, gpu = use_cuda)
    model.classifier = head_patch
    file_name = F"{method}_epoch{num_epochs}_batch{batch_size}_learning_rate{learning_rate}_perm_set_size{perm_set_size}_grid_size{grid_crop_size}_patch_size{patch_crop_size}.tar"

  elif method =="relative_position":
    head_jigsaw =jigsaw.Basic_JigsawHead(1024,perm_set_size, gpu = use_cuda)
    model.classifier = head_jigsaw
    file_name = F"{method}_epoch{num_epochs}_batch{batch_size}_learning_rate{learning_rate}_split{split}.tar"

  elif method== "naive_combination":
    head_patch = patch.Basic_RelativePositionHead(1024, gpu = use_cuda)
    head_jigsaw =jigsaw.Basic_JigsawHead(1024,perm_set_size, gpu = use_cuda)
    file_name = F"{method}_epoch{num_epochs}_batch{batch_size}_learning_rate{learning_rate}_split{split}_perm_set_size{perm_set_size}_grid_size{grid_crop_size}_patch_size{patch_crop_size}.tar"
  
  if from_checkpoint:
    
    checkpoint=torch.load(from_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    plot_loss = checkpoint['loss']
    
    start_i = from_checkpoint.index('epoch')
    end_i =from_checkpoint.index('_batch')
    initial_epoch = int(from_checkpoint[start_i+5:end_i])
    file_name_after_checkpoint = from_checkpoint.replace(from_checkpoint[start_i:end_i],'epoch'+ str( initial_epoch + num_epochs  ))

    if file_name_after_checkpoint.__contains__("split"):
      start_i = from_checkpoint.index('split')
      end_i = from_checkpoint.index('_perm_set') if method == "naive_combination" else from_checkpoint.index('.tar')
      split = float(from_checkpoint[start_i+5: end_i])

    if file_name_after_checkpoint.__contains__("perm_set"):
      start_i = from_checkpoint.index('perm_set')
      end_i = from_checkpoint.index('_grid_size')
      perm_set_size = float(from_checkpoint[start_i+8: end_i])
    

  #Setting permuation_set
  file_name_p_set = F"permutation_set{perm_set_size}.pt"
  PATH_p_set = root_PATH +F"SummerThesis/code/custom_lib/permutation_set/saved_permutation_sets/{file_name_p_set}"
  
  #just ToTensor before patch
  transform_train= transforms.Compose([  transforms.RandomCrop(320), transforms.RandomVerticalFlip(), transforms.ToTensor()])

  #after patch transformation
  transform_after_patching= transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])                                 

  labels_path= root_PATH+"SummerThesis/code/custom_lib/chexpert_load/self_train_labels.pt"
  cheXpert_train_dataset, dataloader = chexpert_load.chexpert_load(root_PATH + "SummerThesis/code/custom_lib/chexpert_load/self_train.csv",
                                                                  transform_train,batch_size, labels_path=labels_path,root_dir = root_dir)

  model=model.to(device=device)
  currentDT = datetime.datetime.now()
  print('START--',file_name)
  for epoch in range(num_epochs):

      for i,  (images, observations) in enumerate(dataloader):   # Load a batch of images with its (index, data, class)
        
        if method == "jigsaw":
          patcher = jigsaw.Jigsaw(images, PATH_p_set, grid_crop_size=grid_crop_size, patch_crop_size=patch_crop_size,
                                        transform =transform_after_patching, gpu = use_cuda, show=show)  
        elif method=="relative_position":
          patcher = patch.Patch(images,split=split,transform=transform_after_patching,show=show)
        
        elif method == "naive_combination":
          if i%2:
            model.classifier = head_jigsaw
            patcher = jigsaw.Jigsaw(images, PATH_p_set, grid_crop_size=grid_crop_size, patch_crop_size=patch_crop_size,
                                            transform =transform_after_patching, gpu = use_cuda, show=show)  
          else:
            model.classifier = head_patch
            patcher = patch.Patch(images,split=split,transform=transform_after_patching,show=show)



        patches, labels =  patcher()
        patches = patches.to(device = device, dtype = torch.float32)
        
        if show:
          print("showa giriyooor",show)
          break
          
        labels = labels.to(device=device, dtype=torch.long)

        #break
        outputs = model(patches)              # Forward pass: compute the output class given a image
        print(outputs.shape)
        print(labels.shape)

        loss = criterion(outputs, labels)           # Compute the loss: difference between the output class and the pre-given label
        optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
        loss.backward()                                   # Backward pass: compute the weight
        optimizer.step()                                  # Optimizer: update the weights of hidden nodes
        

        if i%200 == 0:
                plot_loss.append(loss)
          
                
        if (i+1) % 100 == 0:                              # Logging
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                    %(epoch+1, num_epochs, i+1, len(cheXpert_train_dataset)//batch_size, loss))
        break
  print('training done')

  aftertDT = datetime.datetime.now()
  c=aftertDT-currentDT
  mins,sec=divmod(c.days * 86400 + c.seconds, 60)
  print(mins,"mins ", sec,"secs")

  PATH = file_name_after_checkpoint if from_checkpoint else F"/home/aras/Desktop/saved_models/{file_name}"

  print('END--',PATH)

  if method != "naive_combination":
    torch.save({
              'epoch': initial_epoch + num_epochs if from_checkpoint else  num_epochs ,
              'model_state_dict': model.state_dict(),
              'model_head':,
              'optimizer_state_dict': optimizer.state_dict(),
              'loss':plot_loss}, PATH)
  else:
    torch.save({
              'epoch': initial_epoch + num_epochs if from_checkpoint else  num_epochs ,
              'model_state_dict': model.state_dict(),
              'model_patch_head':head_patch,
              'model_jigsaw_head':head_jigsaw,
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





show=0
learning_rate=0.0001
batch_size=16
resize=320

'''
method="relative_position"
#Params for relative_position
split = 3.0
#Params for jigsaw
grid_crop_size=225
patch_crop_size=64
perm_set_size=300
num_epochs=3
'''
#'''(method="relative_position",num_epochs=3, learning_rate=0.0001, batch_size=16,split = 3.0, grid_crop_size=225,patch_crop_size=64,perm_set_size=300)'''

schedule=[{"method":"relative_position","num_epochs":3},
          {"method":"jigsaw","num_epochs":3},
          {"method":"relative_position","num_epochs":3,"split":2},
          {"method":"naive_combination","num_epochs":3}]

schedule=[{"method":"naive_combination","num_epochs":6,"learning_rate":0.001},
          {"method":"naive_combination","num_epochs":12}]



schedule=[{"method":"relative_position","num_epochs":6,"split":2},
          {"num_epochs":6,"from_checkpoint":"/home/aras/Desktop/saved_models/naive_combination_epoch12_batch16_learning_rate0.0001_split3.0_perm_set_size300_grid_size225_patch_size64.tar"},
          {"num_epochs":6,"from_checkpoint":"/home/aras/Desktop/saved_models/naive_combination_epoch6_batch16_learning_rate0.001_split3.0_perm_set_size300_grid_size225_patch_size64.tar"},
          {"num_epochs":3,"from_checkpoint":"relative_position_epoch3_batch16_learning_rate0.0001_split2.tar"},
          {"num_epochs":3,"from_checkpoint":"relative_position_epoch3_batch16_learning_rate0.0001_split3.tar"}]



schedule=[ {"num_epochs":1,"from_checkpoint":"/home/aras/Desktop/saved_models/relative_position_epoch3_batch16_learning_rate0.0001_split3.0.tar"}]

#schedule=[{"method":"relative_position","num_epochs":1,"split":2}]

for kwargs in schedule:
  self_train(**kwargs)


