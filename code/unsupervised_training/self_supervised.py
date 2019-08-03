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

saved_model_PATH="/vol/bitbucket/ay1218/"

root_PATH_dataset = "/vol/gpudata/ay1218/"
#root_PATH_dataset = saved_model_PATH


root_PATH = "/homes/ay1218/Desktop/"

#coomment out  below for local comp
#root_PATH = "/home/aras/Desktop/"
#root_PATH_dataset = root_PATH
#saved_model_PATH=root_PATH

sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/chexpert_load')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/utilities/load_model')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/selfsupervised_heads/relative_position')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/selfsupervised_heads/rotation')

sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/selfsupervised_heads/jigsaw')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/utilities/plotting_lib')

import chexpert_load
import load_model
import plot_loss_auc_n_precision_recall

use_cuda = True
if use_cuda and torch.cuda.is_available():
    print("using CUDA")
    device = torch.device('cuda:0')
else:
    print("CUDA didn't work, runing in cpu")
    device = torch.device('cpu')


def self_train(method="",num_epochs=3, learning_rate=0.0001, batch_size=16, resize=320 ,K=4, split = 3.0, grid_crop_size=225,patch_crop_size=64,perm_set_size=500 ,from_checkpoint=None ,
 combo=[], root_PATH = root_PATH ,root_PATH_dataset=root_PATH_dataset, saved_model_PATH=saved_model_PATH, show=False,batch_factor=False):


  model = models.densenet121()
  #optimizer =torch.optim.RMSprop(model.parameters(), lr=learning_rate)
  criterion = torch.nn.CrossEntropyLoss().to(device = device)
  plot_loss = {}

  #Setting permuation_set
  PATH_p_set = root_PATH +"SummerThesis/code/custom_lib/utilities/permutation_set/saved_permutation_sets/permutation_set"+ str(perm_set_size)+".pt"
  #just ToTensor before patch
  transform_train= transforms.Compose([  transforms.Resize((resize,resize)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])

  #after patch transformation
  transform_after_patching= transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                               transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  '''
  transform_after_patching= transforms.Compose([ transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  '''

  #constant vars
  kwarg_Jigsaw = { "perm_set_size": perm_set_size, "path_permutation_set":PATH_p_set, "grid_crop_size":grid_crop_size, "patch_crop_size":patch_crop_size, "transform" :transform_after_patching, "gpu": use_cuda, "show":show }
  kwarg_Relative_Position = {"split":split,"transform":transform_after_patching,"show":show,"labels_path":root_PATH}
  kwarg_Rotation = {"K":K,"transform":transform_after_patching,"show":show}

  kwarg_Common ={"num_epochs":num_epochs,"learning_rate":learning_rate, "batch_size":batch_size*batch_factor if batch_factor else batch_size}
  kwargs={"Common": kwarg_Common,"Jigsaw": kwarg_Jigsaw,"Relative_Position": kwarg_Relative_Position,"Rotation":kwarg_Rotation}

  loader = load_model.Load_Model(method=method, combo=combo, from_checkpoint =from_checkpoint, kwargs=kwargs, model=model, plot_loss=plot_loss ,use_cuda=use_cuda )

  file_name , head_arch , plot_loss  = loader()
  n_heads= len(head_arch)

  saved_model_PATH = saved_model_PATH+  "saved_models/self_supervised/"+ file_name[:-4]
  if not os.path.exists(saved_model_PATH): os.mkdir(saved_model_PATH)


  labels_path= root_PATH + "SummerThesis/code/custom_lib/chexpert_load/self_train_labels.pt"
  cheXpert_train_dataset, dataloader = chexpert_load.chexpert_load(root_PATH + "SummerThesis/code/custom_lib/chexpert_load/self_train.csv",
                                            transform_train,batch_size, labels_path=labels_path,root_dir = root_PATH_dataset, num_workers=5)

  print("device", device)
  model=model.to(device=device)
  model.train()
  currentDT = datetime.datetime.now()
  print('START--',file_name)

  for epoch in range(num_epochs):
      for i,  (images, observations,_) in enumerate(dataloader):   # Load a batch of images with its (index, data, class)

        h_id = i % n_heads
        head_dict = head_arch[h_id]
        model.classifier = head_dict["head"]
        patcher = head_dict["patch_func"](image_batch= images,**head_dict["args"])
        optimizer= head_dict['optimizer']

        patches, labels =  patcher()
        patches = patches.to(device = device, dtype = torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        #break
        outputs = model(patches)                          # Forward pass: compute the output class given a image

        loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label



        if batch_factor and i % (batch_factor*n_heads) == h_id:
            optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
        elif not batch_factor:
            optimizer.zero_grad()                             # Intialize the hidden weight to all zeros

        loss.backward()                                   # Backward pass: compute the weight

        if batch_factor and (i+1) % (batch_factor*n_heads) == h_id:
            optimizer.step()
        elif not batch_factor:
            optimizer.step()                                  # Optimizer: update the weights of hidden nodes


        for n in range(n_heads):
          if i%200 == n:
            #print(head_dict['head_name'])
            plot_loss[head_dict['head_name']].append(loss)

            #print(len(plot_loss[head_dict['head_name']]))

        if (i+1) % 100 == 0:                              # Logging
          print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(cheXpert_train_dataset)//batch_size, loss))
          aftertDT = datetime.datetime.now()
          c=aftertDT-currentDT
          mins,sec=divmod(c.days * 86400 + c.seconds, 60)
          print(mins,"mins ", sec,"secs","-----",c.microseconds,"microsec")

        if show:
          print("showa giriyooor",show)
          break

    #    break
    #  break


  print('training done')

  aftertDT = datetime.datetime.now()
  c=aftertDT-currentDT
  mins,sec=divmod(c.days * 86400 + c.seconds, 60)
  print(mins,"mins ", sec,"secs")

  print('END--',file_name)
#'''
  PATH =  saved_model_PATH+"/"+file_name

  head_name_list = [head["head_name"]  for head in head_arch]
  head_state_list = [head["head"].state_dict()  for head in head_arch]
  optimizer_state_list=[head['optimizer'].state_dict()  for head in head_arch]

  torch.save({
             'epoch': kwarg_Common["num_epochs"] ,
             'model_state_dict': model.state_dict(),
             'model_head': dict(zip(head_name_list,head_state_list)),#saving name of the method and the head state
             'optimizer_state_dict': dict(zip(head_name_list, optimizer_state_list)),
             'loss':plot_loss}, PATH)

  curves =plot_loss_auc_n_precision_recall.Curves_AUC_PrecionnRecall(model_name=file_name,root_PATH= saved_model_PATH,mode="just_plot_loss")
  curves.plot_loss(plot_loss=plot_loss)

  #torch.save(model.state_dict(), PATH)
  print('saved  model(model,optim,loss, epoch)')# to google drive')
#'''

#'''(method="relative_position",num_epochs=3, learning_rate=0.0001, batch_size=16,split = 3.0, grid_crop_size=225,patch_crop_size=64,perm_set_size=300)'''

combo = ["Relative_Position","Jigsaw"]

schedule=[{"method":"relative_position","num_epochs":3},
          {"method":"jigsaw","num_epochs":3},
          {"method":"relative_position","num_epochs":3,"split":2},
          {"method":"naive_combination","num_epochs":3}]


combo_RPnJ = ["Relative_Position","Jigsaw"]
combo_RPnR = ["Relative_Position","Rotation"]
combo_RnJ = ["Rotation","Jigsaw"]

combo_all = ["Rotation","Relative_Position","Jigsaw"]

schedule=[ {"num_epochs":1,"from_checkpoint":"/home/aras/Desktop/saved_models/self_supervised/Relative_Position_epoch3_batch16_learning_rate0.0001_split3.0.tar"}]

schedule=[{"method":"Relative_Position","num_epochs":3,"split":3.0}]
schedule=[ {"num_epochs":1,"from_checkpoint":"/home/aras/Desktop/saved_models/self_supervised/Relative_Position_epoch3_batch16_learning_rate0.0001_split3.0.tar"}]


p = saved_model_PATH +'saved_models/self_supervised/'

schedule=[ {"num_epochs":3,"from_checkpoint":p+"Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size500_grid_crop_size225_patch_crop_size64/Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size500_grid_crop_size225_patch_crop_size64.tar"}
          ,{"num_epochs":3,"from_checkpoint":p+"Relative_Position_num_epochs3_batch_size16_learning_rate0.0001_split3.0/Relative_Position_num_epochs3_batch_size16_learning_rate0.0001_split3.0.tar"}
          ,{"num_epochs":3,"from_checkpoint":p+"naive_combination*Relative_Position*Jigsaw*_num_epochs3_batch_size16_learning_rate0.0001_split3.0_perm_set_size500_grid_crop_size225_patch_crop_size64/naive_combination*Relative_Position*Jigsaw*_num_epochs3_batch_size16_learning_rate0.0001_split3.0_perm_set_size500_grid_crop_size225_patch_crop_size64.tar"}]


schedule=[      {"method":"naive_combination","combo":combo,"num_epochs":1,"perm_set_size":100}
                ,{"method":"naive_combination","combo":combo,"num_epochs":1,"perm_set_size":500}
                ,{"method":"Jigsaw","perm_set_size":100,"num_epochs":1}
                ,{"method":"Jigsaw","perm_set_size":500,"num_epochs":1}
                ,{"method":"Relative_Position","split":3.0,"num_epochs":1}
                ,{"num_epochs":1,"from_checkpoint":p+"Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size500_grid_crop_size225_patch_crop_size64/Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size500_grid_crop_size225_patch_crop_size64.tar"}
                ,{"num_epochs":1,"from_checkpoint":p+"Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size500_grid_crop_size225_patch_crop_size64/Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size500_grid_crop_size225_patch_crop_size64.tar"}
                ,{"from_checkpoint":p+"Relative_Position_num_epochs3_batch_size16_learning_rate0.0001_split3.0/Relative_Position_num_epochs3_batch_size16_learning_rate0.0001_split3.0.tar"}
                ,{"from_checkpoint":p+"naive_combination*Relative_Position*Jigsaw*_num_epochs3_batch_size16_learning_rate0.0001_split3.0_perm_set_size500_grid_crop_size225_patch_crop_size64/naive_combination*Relative_Position*Jigsaw*_num_epochs3_batch_size16_learning_rate0.0001_split3.0_perm_set_size500_grid_crop_size225_patch_crop_size64.tar"}]

#schedule=[ {"from_checkpoint":p+"naive_combination*Relative_Position*Jigsaw*_num_epochs3_batch_size16_learning_rate0.0001_split3.0_perm_set_size500_grid_crop_size225_patch_crop_size64/naive_combination*Relative_Position*Jigsaw*_num_epochs3_batch_size16_learning_rate0.0001_split3.0_perm_set_size500_grid_crop_size225_patch_crop_size64.tar"}]
schedule=[      {"method":"naive_combination","combo":combo,"num_epochs":2,"perm_set_size":100}
                ,{"method":"naive_combination","combo":combo,"num_epochs":2,"perm_set_size":500}
                ,{"method":"Jigsaw","perm_set_size":100,"num_epochs":1}
                ,{"method":"Jigsaw","perm_set_size":500,"num_epochs":1}
                ,{"method":"Relative_Position","split":3.0,"num_epochs":1}]

schedule = [{"method":"Rotation","num_epochs":1,"batch_size":4,"batch_factor":4}

, {"method":"naive_combination","combo":combo_RPnR,"num_epochs":2,"batch_size":4,"batch_factor":4}

 ,{"method":"naive_combination","combo":combo_RnJ,"num_epochs":2,"perm_set_size":100,"batch_size":4,"batch_factor":4}
 , {"method":"naive_combination","combo":combo_RnJ,"num_epochs":2,"perm_set_size":500,"batch_size":4,"batch_factor":4}

 , {"method":"naive_combination","combo":combo_all,"num_epochs":3,"perm_set_size":100,"batch_size":4,"batch_factor":4}
 , {"method":"naive_combination","combo":combo_all,"num_epochs":3,"perm_set_size":500,"batch_size":4,"batch_factor":4}]


import time
#min = 60
#time.sleep(2*60*min)

for kwargs in schedule:
  self_train(**kwargs)
