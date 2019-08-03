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
import time
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
#saved_model_PATH=root_PATH

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

def transfer_learning(  num_epochs=3, resize= 320, batch_size=16,posw =1, data_rate=1,pre_trained=False, pre_trained_PATH="", from_checkpoint="", root_PATH = root_PATH,learning_rate=0.0001,num_workers=5,
                                                        root_PATH_dataset=root_PATH_dataset, saved_model_PATH=saved_model_PATH):

    #after patch transformation
    transform= transforms.Compose([             transforms.Resize((resize,resize)),transforms.ToTensor(),
                                                transforms.Lambda(lambda x: torch.cat([x, x, x], 0))
                                                ])#,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    model=models.densenet121(pretrained=pre_trained,num_classes = 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if posw:
        criterion =nn.BCEWithLogitsLoss(pos_weight=chexpert_load.load_posw()).to(device=device)
    else:
        criterion =nn.BCEWithLogitsLoss().to(device=device)

    plot_loss = []

    kwarg_Common ={"num_epochs":num_epochs,"learning_rate":learning_rate,"batch_size":batch_size}
    kwargs={"Common":kwarg_Common}


    if pre_trained_PATH or from_checkpoint :
        loader = load_model.Load_Model(method="TL",pre_trained = pre_trained_PATH,from_checkpoint = from_checkpoint, kwargs=kwargs, model=model,  plot_loss=plot_loss, use_cuda=use_cuda  )
        file_name , optimizer ,plot_loss  = loader()

    else:
        print("training from scratch ")
        file_name = "from_scratch_epoch"+str(num_epochs)+"_batch"+str(batch_size)+"_learning_rate"+str(learning_rate)+".tar"

    file_name= "data_rate"+str(data_rate)+"_"+file_name if data_rate != 1 else file_name
    file_name= "pre_trainedIMAGENET_"+file_name if  pre_trained else file_name
    file_name= "no_posw_"+file_name if not posw else file_name


    saved_model_PATH = saved_model_PATH+"saved_models/transfer_learning/"+file_name[:-4]
    if not os.path.exists(saved_model_PATH): os.mkdir(saved_model_PATH)

    labels_path = root_PATH+"SummerThesis/code/custom_lib/chexpert_load/labels.pt"
    cheXpert_train_dataset, dataloader = chexpert_load.chexpert_load(root_PATH +"SummerThesis/code/custom_lib/chexpert_load/train.csv",transform, kwarg_Common["batch_size"],
                                                                        num_workers=num_workers, data_rate =data_rate, labels_path = labels_path, root_dir= root_PATH_dataset )

    currentDT = datetime.datetime.now()
    model=model.to(device=device)
    print("started training")
    print('START--',file_name)
    model.train()
    for epoch in range(num_epochs):
        for i,  (images, labels,_) in enumerate(dataloader):   # Load a batch of images with its (index, data, class)

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
            #break
        #break

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

schedule=[  {"transfer_learning":0}]

c = saved_model_PATH +'saved_models/transfer_learning/'
p = saved_model_PATH +'saved_models/self_supervised/'
s =  saved_model_PATH +'saved_models/semi_supervised/'

#schedule=[  { "from_checkpoint":c+"from_scratch_epoch3_batch16_learning_rate0.0001/from_scratch_epoch3_batch16_learning_rate0.0001.tar"}    ]


schedule=[

        {"pre_trained_PATH":s+"CC_GAN2_num_epochs3_learning_rate0.0002_batch_size16_resize128_label_rate0.2/CC_GAN2_num_epochs3_learning_rate0.0002_batch_size16_resize128_label_rate0.2.tar"}
        ,{"pre_trained_PATH": s+"CC_GAN2_num_epochs3_learning_rate0.0002_batch_size16_resize256_label_rate0.2/CC_GAN2_num_epochs3_learning_rate0.0002_batch_size16_resize256_label_rate0.2.tar"}
        ,{"pre_trained_PATH": s+"CC_GAN2_num_epochs3_learning_rate0.0002_batch_size32_resize128_label_rate0.2/CC_GAN2_num_epochs3_learning_rate0.0002_batch_size32_resize128_label_rate0.2.tar"}
        ,{"pre_trained_PATH": s+"CC_GAN2_num_epochs3_learning_rate0.0002_batch_size64_resize128_label_rate0.2/CC_GAN2_num_epochs3_learning_rate0.0002_batch_size64_resize128_label_rate0.2.tar"}
        ]




schedule=[

        {"data_rate":0.2,"pre_trained_PATH":p+"Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size100_grid_crop_size225_patch_crop_size64/Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size100_grid_crop_size225_patch_crop_size64.tar"}
        ,{"data_rate":0.2,"pre_trained_PATH": p+"Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size500_grid_crop_size225_patch_crop_size64/Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size500_grid_crop_size225_patch_crop_size64.tar"}
        ,{"data_rate":0.2,"pre_trained_PATH": p+"Jigsaw_num_epochs6_batch_size16_learning_rate0.0001_perm_set_size100_grid_crop_size225_patch_crop_size64/Jigsaw_num_epochs6_batch_size16_learning_rate0.0001_perm_set_size100_grid_crop_size225_patch_crop_size64.tar"}
        ,{"data_rate":0.2,"pre_trained_PATH": p+"Jigsaw_num_epochs6_batch_size16_learning_rate0.0001_perm_set_size500_grid_crop_size225_patch_crop_size64/Jigsaw_num_epochs6_batch_size16_learning_rate0.0001_perm_set_size500_grid_crop_size225_patch_crop_size64.tar"}
        ,{"data_rate":0.2,"pre_trained_PATH": p+"/.tar"}
        ,{"data_rate":0.2,"pre_trained_PATH": p+"/.tar"}
        ,{"data_rate":0.2,"pre_trained_PATH": p+"/.tar"}
        ]

schedule=[{"num_epochs":3}
          ,{"num_epochs":3,"pre_trained":True}
          ,{"num_epochs":3,"pre_trained":True,"posw":False}]
#min = 60
#time.sleep(4*60*min)


for kwargs in schedule:

    transfer_learning(**kwargs)
