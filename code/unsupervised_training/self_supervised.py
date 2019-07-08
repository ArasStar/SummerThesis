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
save_model_path="/vol/bitbucket/ay1218/"
root_PATH_dataset = "/vol/gpudata/ay1218/"
root_PATH = "/home/aras/Desktop/"

#root_PATH_dataset = root_PATH
#save_model_path=root_PATH

sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/chexpert_load')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/relative_position')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/jigsaw')

import chexpert_load
import relative_position
import jigsaw

head_libs={"Relative_Position":relative_position ,"Jigsaw":jigsaw}

use_cuda = True
if use_cuda and torch.cuda.is_available():
    print("using CUDA")
    device = torch.device('cuda:0')
else:
    print("CUDA didn't work")
    device = torch.device('cpu')


#print(os.getcwd())
#root_dir = '/../../'
#root_dir = '/../'

def self_train(method="Relative_Position",num_epochs=3, learning_rate=0.0001, batch_size=16, split = 3.0, grid_crop_size=225,patch_crop_size=64,perm_set_size=300 ,
                  from_checkpoint=None , combo=["Relative_Position","Jigsaw"], root_PATH = root_PATH ,root_PATH_dataset=root_PATH_dataset,save_model_path=save_model_path, show=False):

  #Setting permuation_set
  file_name_p_set = "permutation_set"+ str(perm_set_size)+".pt"
  PATH_p_set = root_PATH +"SummerThesis/code/custom_lib/permutation_set/saved_permutation_sets/"+file_name_p_set
  out_D ={"Relative_Position":8 ,"Jigsaw":perm_set_size}

  #just ToTensor before patch
  transform_train= transforms.Compose([  transforms.RandomCrop(320), transforms.RandomVerticalFlip(), transforms.ToTensor()])

  #after patch transformation
  transform_after_patching= transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  model = models.densenet121()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  criterion = torch.nn.CrossEntropyLoss().to(device = device)
  plot_loss = []

  if from_checkpoint:
        if from_checkpoint.__contains__("combination"):
          method = from_checkpoint[from_checkpoint.index("_supervised/")+12: from_checkpoint.index('combination')+11]
          combo = from_checkpoint[from_checkpoint.index("**")+2:from_checkpoint.index("**_epoch")].split("**")
        else:
          method = from_checkpoint[from_checkpoint.index("_supervised/")+12:from_checkpoint.index('_epoch')]
        #NEED RTO DO SEPERATE THINGS FOR COmBO anD ALONE


  if not method.__contains__("combination"):

    head_task = head_libs[method].Basic_Head(1024, out_D[method], gpu = use_cuda)
    model.classifier = head_task
    to_patch = getattr(head_libs[method],method)

    if method=="Jigsaw":
      file_name = method+"_epoch"+str(num_epochs)+"_batch"+str(batch_size)+"_learning_rate"+str(learning_rate)+"_perm_set_size"+str(perm_set_size)+"_grid_size"+str(grid_crop_size)+"_patch_size"+str(patch_crop_size)+".tar"
      kwarg = { "path_permutation_set":PATH_p_set, "grid_crop_size":grid_crop_size, "patch_crop_size":patch_crop_size, "transform" :transform_after_patching, "gpu": use_cuda, "show":show }

    elif method == "Relative_Position":
      file_name = method+"_epoch" + str(num_epochs)+ "_batch" +str(batch_size)+"_learning_rate"+str(learning_rate)+"_split"+str(split)+".tar"
      kwarg = {"split":split,"transform":transform_after_patching,"show":show}

  elif method == "naive_combination":

    file_name = method+"**"+"**".join(combo)+"**_epoch"+str(num_epochs)+"_batch"+str(batch_size)+"_learning_rate"+str(learning_rate)+"_split"+str(split)+"_perm_set_size"+str(perm_set_size)+"_grid_size"+str(grid_crop_size)+"_patch_size"+str(patch_crop_size)+".tar"
    n_heads = len(combo)
    heads = []
    for h in combo:

      head_module = head_libs[h]
      method_head = head_module.Basic_Head(1024,out_D[h], gpu = use_cuda)
      to_patch = getattr(head_module,h)

      if h == "Relative_Position":
        kwarg = {"split":split,"transform":transform_after_patching,"show":show}

      elif h =="Jigsaw":
        kwarg = { "path_permutation_set":PATH_p_set, "grid_crop_size":grid_crop_size, "patch_crop_size":patch_crop_size, "transform" :transform_after_patching, "gpu": use_cuda, "show":show }

      heads.append((method_head, to_patch, kwarg, h))

  if from_checkpoint:

    #loading model
    checkpoint = torch.load(from_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)#loading features
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for state in optimizer.state.values():
      for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.to(device=device)

    plot_loss = checkpoint['loss']

    if from_checkpoint.__contains__("combination"):
      head_dict =  checkpoint["model_heads"] # dict headname: headstateDict

      for head in heads:
        h_name = head[-1]
        head[0].load_state_dict(head_dict[h_name])

    else:
      head_task.load_state_dict(checkpoint['model_head'])

    #aranging names and params
    start_i = from_checkpoint.index('epoch')
    end_i =from_checkpoint.index('_batch')
    initial_epoch = int(from_checkpoint[start_i+5:end_i])
    file_name_after_checkpoint = from_checkpoint.replace(from_checkpoint[start_i:end_i],'epoch'+ str( initial_epoch + num_epochs))

    if file_name_after_checkpoint.__contains__("split"):
      start_i = from_checkpoint.index('split')
      end_i = from_checkpoint.index('_perm_set') if method == "naive_combination" else from_checkpoint.index('.tar')
      split = float(from_checkpoint[start_i+5: end_i])

    if file_name_after_checkpoint.__contains__("perm_set"):
      start_i = from_checkpoint.index('set_size')
      end_i = from_checkpoint.index('_grid_size')
      perm_set_size = float(from_checkpoint[start_i+8: end_i])

    file_name=file_name_after_checkpoint



  labels_path= root_PATH + "SummerThesis/code/custom_lib/chexpert_load/self_train_labels.pt"
  cheXpert_train_dataset, dataloader = chexpert_load.chexpert_load(root_PATH + "SummerThesis/code/custom_lib/chexpert_load/self_train.csv",
                                                                  transform_train,batch_size, labels_path=labels_path,root_dir = root_PATH_dataset)

  model=model.to(device=device)
  model.train()
  currentDT = datetime.datetime.now()
  print('START--',file_name)
  for epoch in range(num_epochs):

      for i,  (images, observations) in enumerate(dataloader):   # Load a batch of images with its (index, data, class)

        if  not method.__contains__("combination"):
          patcher = to_patch(image_batch=images,**kwarg)

        else:
          model_set = heads[i % n_heads]
          model.classifier = model_set[0]
          patcher = model_set[1](image_batch= images,**model_set[2])

        patches, labels =  patcher()
        patches = patches.to(device = device, dtype = torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        #break
        outputs = model(patches)                          # Forward pass: compute the output class given a image

        loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
        optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
        loss.backward()                                   # Backward pass: compute the weight
        optimizer.step()                                  # Optimizer: update the weights of hidden nodes

        if i%200 == 0:
          plot_loss.append(loss)

        if (i+1) % 100 == 0:                              # Logging
          print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(cheXpert_train_dataset)//batch_size, loss))
          aftertDT = datetime.datetime.now()
          c=aftertDT-currentDT
          mins,sec=divmod(c.days * 86400 + c.seconds, 60)
          print(mins,"mins ", sec,"secs")

        if show:
          print("showa giriyooor",show)
          break

  print('training done')

  aftertDT = datetime.datetime.now()
  c=aftertDT-currentDT
  mins,sec=divmod(c.days * 86400 + c.seconds, 60)
  print(mins,"mins ", sec,"secs")

  PATH = file_name_after_checkpoint if from_checkpoint else save_model_path+"saved_models/self_supervised/"+file_name

  print('END--',PATH)


  if method != "naive_combination":
    torch.save({
              'epoch': initial_epoch + num_epochs if from_checkpoint else  num_epochs ,
              'model_state_dict': model.state_dict(),
              'model_head': model.classifier.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss':plot_loss}, PATH)
  else:
    head_state_list = [head[0].state_dict()  for head in heads]
    head_name_list = [head[-1]  for head in heads]
    torch.save({
              'epoch': initial_epoch + num_epochs if from_checkpoint else  num_epochs ,
              'model_state_dict': model.state_dict(),
              'model_heads': dict(zip(head_name_list,head_state_list)),#saving name of the method and the head state
              'optimizer_state_dict': optimizer.state_dict(),
              'loss':plot_loss}, PATH)

  #torch.save(model.state_dict(), PATH)
  print('saved  model(model,optim,loss, epoch)')# to google drive')


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



schedule=[ {"num_epochs":1,"from_checkpoint":"/home/aras/Desktop/saved_models/self_supervised/Relative_Position_epoch3_batch16_learning_rate0.0001_split3.0.tar"}]

schedule=[{"method":"Relative_Position","num_epochs":3,"split":3.0}]
schedule=[ {"num_epochs":1,"from_checkpoint":"/home/aras/Desktop/saved_models/self_supervised/Relative_Position_epoch3_batch16_learning_rate0.0001_split3.0.tar"}]

schedule=[{"method":"Jigsaw","num_epochs":3}]
schedule=[{"method":"Relative_Position","num_epochs":3}]
schedule =[{"method":"naive_combination","num_epochs":3}]


for kwargs in schedule:
  self_train(**kwargs)
