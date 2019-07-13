# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import os#for CUDA tracking
#from __future__ import print_function, division
import torch
import pandas as pd

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
#root_PATH_dataset = saved_model_PATH


root_PATH = "/homes/ay1218/Desktop/"

#comment out below for local comp

#root_PATH = "/home/aras/Desktop/"
#root_PATH_dataset = root_PATH
saved_model_PATH=root_PATH

sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/chexpert_load')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/utilities/load_model')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/selfsupervised_heads/relative_position')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/selfsupervised_heads/jigsaw')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/utilities/plotting_lib')

import chexpert_load
import load_model
import plot_loss_auc_n_precision_recall
import validation



use_cuda = True
if use_cuda and torch.cuda.is_available():
    print("using CUDA")
    device = torch.device('cuda:0')
else:
    print("CUDA didn't work")
    device = torch.device('cpu')

def transfer_learning(  num_epochs=3, resize= 320, batch_size=16, pre_trained_PATH="", from_checkpoint="", root_PATH = root_PATH,
                                                        root_PATH_dataset=root_PATH_dataset, saved_model_PATH=saved_model_PATH):


    learning_rate=0.0001
    #after patch transformation
    transform= transforms.Compose([             transforms.Resize((resize,resize)),transforms.ToTensor(),
                                                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    model=models.densenet121(num_classes = 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion =nn.BCEWithLogitsLoss(pos_weight=chexpert_load.load_posw()).to(device=device)
    plot_loss = []

    kwarg_Common ={"num_epochs":num_epochs,"learning_rate":learning_rate,"batch_size":batch_size}
    kwargs={"Common":kwarg_Common}


    if pre_trained_PATH :
        loader = load_model.Load_Model(method="TL",pre_trained = pre_trained_PATH, kwargs=kwargs, model=model, optimizer=optimizer, plot_loss=plot_loss  )
        file_name , optimizer ,plot_loss  = loader()

    elif from_checkpoint :
        loader = load_model.Load_Model(method="TL",from_checkpoint = from_checkpoint, kwargs=kwargs, model=model, optimizer=optimizer, plot_loss=plot_loss  )
        file_name , optimizer , plot_loss  = loader()

    else:
        print("training from scratch")
        file_name = "from_scratch_epoch"+str(num_epochs)+"_batch"+str(batch_size)+"_learning_rate"+str(learning_rate)+".tar"


    saved_model_PATH = saved_model_PATH+"saved_models/transfer_learning/"+file_name[:-4]
    if not os.path.exists(saved_model_PATH): os.mkdir(saved_model_PATH)

    labels_path = root_PATH+"SummerThesis/code/custom_lib/chexpert_load/labels.pt"
    cheXpert_train_dataset, dataloader = chexpert_load.chexpert_load(root_PATH +"SummerThesis/code/custom_lib/chexpert_load/train.csv",transform,
                                                                    kwarg_Common["batch_size"],labels_path = labels_path,root_dir= root_PATH_dataset )

    print(len(plot_loss))
    currentDT = datetime.datetime.now()
    model=model.to(device=device)

    print("started training")
    print('START--',file_name )

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

    aftertDT = datetime.datetime.now()
    c = aftertDT - currentDT
    mins,sec = divmod(c.days * 86400 + c.seconds, 60)
    print(mins,"mins ", sec,"secs")
    print('END--',file_name)

    # Calculating valid error plotting AUC , Precisinon -Recall , plot loss , saving figures, printingg auc differences
    PATH =  saved_model_PATH+"/"+file_name
    val = validation.Validation(chexpert_load=chexpert_load,model=model, plot_loss=plot_loss, bs = 16, root_PATH = root_PATH, root_PATH_dataset=root_PATH_dataset, saved_model_PATH = saved_model_PATH, file_name=file_name, gpu=use_cuda )
    val()

    torch.save({'epoch':  num_epochs ,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':plot_loss}, PATH)
#FINIIIIIISH
schedule=[  {"transfer_learning":1,"pre_trained_PATH":"/home/aras/Desktop/saved_models/naive_combination_epoch12_batch16_learning_rate0.0001_split3.0_perm_set_size300_grid_size225_patch_size64.tar"},
            {"transfer_learning":1,"pre_trained_PATH":"/home/aras/Desktop/saved_models/jigsaw_epoch3_batch16_learning_rate0.0001_perm_set_size300_grid_size225_patch_size64.tar"},
            {"transfer_learning":1,"pre_trained_PATH":"/home/aras/Desktop/saved_models/relative_position_epoch3_batch16_learning_rate0.0001_split3.0.tar"}]

schedule=[  {"transfer_learning":0}]

p = saved_model_PATH +'saved_models/transfer_learning/'

schedule=[  { "from_checkpoint":p+"from_scratch_epoch3_batch16_learning_rate0.0001/from_scratch_epoch3_batch16_learning_rate0.0001.tar"}    ]

for kwargs in schedule:
    transfer_learning(**kwargs)
