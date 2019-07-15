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
saved_model_PATH=root_PATH

sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/chexpert_load')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/utilities/load_model')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/utilities/plotting_lib')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/semi_supervised_CC_GAN')

import chexpert_load
import load_model
import cc-gan

use_cuda = True
if use_cuda and torch.cuda.is_available():
    print("using CUDA")
    device = torch.device('cuda:0')
else:
    print("CUDA didn't work")
    device = torch.device('cpu')


def train_ccgan(method="CC-GAN"num_epochs=3, lr=0.0002, batch_size=16, from_checkpoint=None, from_Pretrained = None ,root_PATH = root_PATH ,root_PATH_dataset=root_PATH_dataset, saved_model_PATH=saved_model_PATH, show=False):

    kwarg_Common ={"num_epochs":num_epochs,"learning_rate":learning_rate,"batch_size":batch_size}
    kwargs={"Common":kwarg_Common}
    plot_loss = [[],[]]

    #just ToTensor before patch
    transform_train= transforms.Compose([ transforms.RandomCrop(320), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    '''
    #after patch transformation
    transform_after_patching= transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                                 transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    '''

    labels_path= root_PATH + "SummerThesis/code/custom_lib/chexpert_load/labels.pt"
    cheXpert_train_dataset, dataloader = chexpert_load.chexpert_load(root_PATH + "SummerThesis/code/custom_lib/chexpert_load/train.csv",
                                                                     transform_train,batch_size, labels_path=labels_path,root_dir = root_PATH_dataset)
    netD = Discriminator()
    netG = Generator()
    netD.to(device)
    netG.to(device)

    advs_criterion = nn.BCELoss().to(device)
    classification_criterion = nn.BCEWithLogitsLoss(pos_weight=chexpert_load.load_posw()).to(device=device)


    #apply init weight???
    beta1=0.5
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    real_label = 1
    fake_label = 0

    loader = load_model.Load_Model(method=method, combo=combo, from_checkpoint =from_checkpoint, kwargs=kwargs, model=model, plot_loss=plot_loss  )
    file_name , plot_loss  = loader()
    saved_model_PATH = saved_model_PATH+  "saved_models/semi_supervised/"+ file_name[:-4]
    if not os.path.exists(saved_model_PATH): os.mkdir(saved_model_PATH)

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    currentDT = datetime.datetime.now()

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, real_images , class_labels in enumerate(dataloader):

            real_images = real_images.to(device=device,dtype=torch.float)
            class_labels = class_labels.to(device=device,dtype=torch.float)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            label = torch.full((batch_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_images).view(-1)
            # Calculate loss on all-real batch
            errD_real = advs_criterion(output[0], label) + classification_criterion(output[1:], class_labels)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors (context conditioned in our case)
            patcher = cc-gan.Patcher_CC_GAN(real_images)
            context_conditioned,low_res ,cord = patcher()
            # Generate fake image batch with G
            fake = netG(context_conditioned,low_res,cord)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake =  advs_criterion(output[0], label) + classification_criterion(output[1:], class_labels)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = advs_criterion(output[0], label) + classification_criterion(output[1:], class_labels)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 100 == 0:
                 aftertDT = datetime.datetime.now()
                 c=aftertDT-currentDT
                 mins,sec=divmod(c.days * 86400 + c.seconds, 60)
                 print(mins,"mins ", sec,"secs")

                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if i % 200 == 0:
                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):

                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    print('training done')
    aftertDT = datetime.datetime.now()
    c=aftertDT-currentDT
    mins,sec=divmod(c.days * 86400 + c.seconds, 60)
    print(mins,"mins ", sec,"secs")
    print('END--',file_name)

    PATH =  saved_model_PATH+"/"+file_name

    head_name_list = [head["head_name"]  for head in head_arch]
    head_state_list = [head["head"].state_dict()  for head in head_arch]
    optimizer_state_list=[head['optimizer'].state_dict()  for head in head_arch]

    torch.save({
                'epoch': kwarg_Common["num_epochs"] ,
                'G_model_state_dict': netG.state_dict(),
                'G_optimizer_state_dict': optimizerG.step ,
                'G_loss':plot_loss,
                'D_model_state_dict': netD.state_dict(),
                'D_optimizer_state_dict': optimizerD.step,
                'D_loss':plot_loss}, PATH)

     #torch.save(model.state_dict(), PATH)



     print('saved  model(model,optim,loss, epoch)')

     plt.figure(figsize=(15,5))
     plt.title("Generator and Discriminator Loss During Training")
     plt.plot(G_losses,label="G")
     plt.plot(D_losses,label="D")
     plt.xlabel("iterations")
     plt.ylabel("Loss")
     plt.legend()
     plt.savefig(PATH+'/'+'plot_loss_'+ '.png')






#'''(method="relative_position",num_epochs=3, learning_rate=0.0001, batch_size=16,split = 3.0, grid_crop_size=225,patch_crop_size=64,perm_set_size=300)'''


p = saved_model_PATH +'saved_models/semi_supervised/'
schedule=[{"method":"Jigsaw","num_epochs":3},
          {"method":"Relative_Position","num_epochs":3},
          {"method":"naive_combination","combo":combo,"num_epochs":3},
          {"num_epochs":3,"from_checkpoint":p+"Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size300_grid_crop_size225_patch_crop_size64/Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size300_grid_crop_size225_patch_crop_size64.tar"}
          ,{"num_epochs":3,"from_checkpoint":p+"Relative_Position_num_epochs3_batch_size16_learning_rate0.0001_split3.0/Relative_Position_num_epochs3_batch_size16_learning_rate0.0001_split3.0.tar"}
          ,{"num_epochs":3,"from_checkpoint":p+"naive_combination*Relative_Position*Jigsaw*_num_epochs3_batch_size16_learning_rate0.0001_split3.0_perm_set_size300_grid_crop_size225_patch_crop_size64/naive_combination*Relative_Position*Jigsaw*_num_epochs3_batch_size16_learning_rate0.0001_split3.0_perm_set_size300_grid_crop_size225_patch_crop_size64.tar"}]

schedule=[ {"num_epochs":3,"from_checkpoint":p+"Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size500_grid_crop_size225_patch_crop_size64/Jigsaw_num_epochs3_batch_size16_learning_rate0.0001_perm_set_size500_grid_crop_size225_patch_crop_size64.tar"}
          ,{"num_epochs":3,"from_checkpoint":p+"Relative_Position_num_epochs3_batch_size16_learning_rate0.0001_split3.0/Relative_Position_num_epochs3_batch_size16_learning_rate0.0001_split3.0.tar"}
          ,{"num_epochs":3,"from_checkpoint":p+"naive_combination*Relative_Position*Jigsaw*_num_epochs3_batch_size16_learning_rate0.0001_split3.0_perm_set_size500_grid_crop_size225_patch_crop_size64/naive_combination*Relative_Position*Jigsaw*_num_epochs3_batch_size16_learning_rate0.0001_split3.0_perm_set_size500_grid_crop_size225_patch_crop_size64.tar"}]


for kwargs in schedule:
  train_ccgan(**kwargs)











































#
