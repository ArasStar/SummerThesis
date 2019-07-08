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
print("hooop")

saved_model_PATH="/vol/bitbucket/ay1218/"
root_PATH_dataset = "/vol/gpudata/ay1218/"
root_PATH = "/homes/ay1218/Desktop/"
#root_PATH = "/home/aras/Desktop/"
#root_PATH_dataset = root_PATH

sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/plotting_lib')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/chexpert_load')
import plot_loss_auc_n_precision_recall
import chexpert_load

use_cuda = True
if use_cuda and torch.cuda.is_available():
    print("using CUDA")
    device = torch.device('cuda:0')
else:
    print("CUDA didn't work")
    device = torch.device('cpu')

def transfer_learning(  num_epochs=3, resize= 320, batch_size=8, pre_trained_PATH="", root_PATH = root_PATH, root_PATH_dataset=root_PATH_dataset, saved_model_PATH=saved_model_PATH):

    learning_rate=0.0001
    #after patch transformation
    transform= transforms.Compose([             transforms.Resize((resize,resize)),transforms.ToTensor(),
                                                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    labels_path = root_PATH+"SummerThesis/code/custom_lib/chexpert_load/labels.pt"
    cheXpert_train_dataset, dataloader = chexpert_load.chexpert_load(root_PATH +"SummerThesis/code/custom_lib/chexpert_load/train.csv",transform,
                                                                    batch_size,labels_path = labels_path,root_dir= root_PATH_dataset )


    model=models.densenet121(num_classes = 5)

    if pre_trained_PATH:
        print("tranfering the weights")
        checkpoint = torch.load(pre_trained_PATH)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False) # just features get downloaded classifier stays
        splited = pre_trained_PATH.split('/')
        saved_model_PATH = saved_model_PATH+"saved_models/trasfer_learning/"+splited[-1][:-4]
        if not os.path.exists(saved_model_PATH): os.mkdir(saved_model_PATH)
        file_path = saved_model_PATH + "/TL_epoch"+str(num_epochs)+"_batch"+str(batch_size)+"_learning_rate"+str(learning_rate)+"---" + splited[-1]
    else:
        print("training from scratch")
        saved_model_PATH = saved_model_PATH +"saved_models/transfer_learning/from_scratch__epoch"+str(num_epochs)+"_batch"+str(batch_size)+"_learning_rate"+str(learning_rate)

        if not os.path.exists(saved_model_PATH): os.mkdir(saved_model_PATH)
        splited = saved_model_PATH.split('/')[-1]
        file_path = saved_model_PATH + splited + ".tar"

    model=model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion =nn.BCEWithLogitsLoss(pos_weight=chexpert_load.load_posw()).to(device=device)
    plot_loss = []

    currentDT = datetime.datetime.now()

    print("started training")
    print('START--',"TL_"+file_path.split("/")[-1] )

    model.train()
    for epoch in range(num_epochs):
        for i,  (images, labels) in enumerate(dataloader):   # Load a batch of images with its (index, data, class)

            images = images.to(device=device,dtype=torch.float)
            labels = labels.to(device=device,dtype=torch.float)

            outputs = model(images).to(device=device)   # Forward pass: compute the output class given a image

            loss = criterion(outputs, labels)           # Compute the loss: difference between the output class and the pre-given label

            optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
            loss.backward()                                   # Backward pass: compute the weight
            optimizer.step()                                  # Optimizer: update the weights of hidden nodes

            if (i+1) % 100 == 0:                              # Logging
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                    %(epoch+1, num_epochs, i+1, len(cheXpert_train_dataset)//batch_size, loss))
                aftertDT = datetime.datetime.now()
                c=aftertDT-currentDT
                mins,sec = divmod(c.days * 86400 + c.seconds, 60)
                print(mins,"mins ", sec,"secs")

            if i % 200 == 0:
                plot_loss.append(loss)
            #DELETEEEEE
            break
        break

    print('training done')

    aftertDT = datetime.datetime.now()
    c=aftertDT-currentDT
    mins,sec = divmod(c.days * 86400 + c.seconds, 60)
    print(mins,"mins ", sec,"secs")
    print('END--',"TL_"+file_path.split("/")[-1])



    #VALIDATION SET
    cheXpert_valid_dataset, valid_dataloader = chexpert_load.chexpert_load(root_PATH+"SummerThesis/code/custom_lib/chexpert_load/valid.csv",transform,
                                                                 batch_size, shuffle=False,root_dir=root_PATH_dataset)

    transform= transforms.Compose([             transforms.Resize((resize,resize)),transforms.ToTensor(),
                                                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    probs= []
    probs=np.array(probs)
    acts= []
    acts=np.array(acts)

    model.eval()
    with torch.no_grad():
        for images, labels in valid_dataloader:

            images = images.to(device = device)
            labels = labels.to(device = device)
            outputs = model(images)

            #sigmoid gets rounded
            probabilities= torch.sigmoid(outputs)
            #predictions=probs.round()

            probs = probabilities.cpu().numpy()  if probs.size ==0 else np.vstack((probs,probabilities.cpu().numpy()))
            acts = labels.cpu().numpy()  if acts.size ==0 else np.vstack((acts,labels.cpu().numpy()))

    # SAVING PLOTS and models
    curves =plot_loss_auc_n_precision_recall.Curves_AUC_PrecionnRecall(chexpert_load,cheXpert_valid_dataset, probs, acts, model_name="TL_"+ file_path.split("/")[-2],root_PATH= saved_model_PATH)
    curves()
    curves.plot_loss(plot_loss)

    print('file_path', file_path)
    torch.save({'epoch':  num_epochs ,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':plot_loss}, file_path)
    print("finished eval , saved model and plots")
    curves.auc_difference_print()

#FINIIIIIISH
schedule=[  {"transfer_learning":1,"pre_trained_PATH":"/home/aras/Desktop/saved_models/naive_combination_epoch12_batch16_learning_rate0.0001_split3.0_perm_set_size300_grid_size225_patch_size64.tar"},
            {"transfer_learning":1,"pre_trained_PATH":"/home/aras/Desktop/saved_models/jigsaw_epoch3_batch16_learning_rate0.0001_perm_set_size300_grid_size225_patch_size64.tar"},
            {"transfer_learning":1,"pre_trained_PATH":"/home/aras/Desktop/saved_models/relative_position_epoch3_batch16_learning_rate0.0001_split3.0.tar"}]

schedule=[  {"transfer_learning":0},
            {"transfer_learning":1,"pre_trained_PATH":"/home/aras/Desktop/saved_models/naive_combination_epoch12_batch16_learning_rate0.0001_split3.0_perm_set_size300_grid_size225_patch_size64.tar"}]


schedule=[
            { "batch_size": 8}]

for kwargs in schedule:
    transfer_learning(**kwargs)
