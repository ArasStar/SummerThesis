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
import torchvision.utils as vutils
#MODELS
import re
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib.animation as animation
import datetime
import time
import sys
from PIL import Image
from IPython.display import HTML

torch.autograd.set_detect_anomaly(True)
saved_model_PATH="/vol/bitbucket/ay1218/"

root_PATH_dataset = "/vol/gpudata/ay1218/"
#root_PATH_dataset = saved_model_PATH

root_PATH = "/homes/ay1218/Desktop/"

#coomment out  below for local comp
#root_PATH = "/home/aras/Desktop/"
#root_PATH_dataset = root_PATH
#saved_model_PATH=root_PATH

sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/selfsupervised_heads/relative_position')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/selfsupervised_heads/jigsaw')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/chexpert_load')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/utilities/load_model')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/utilities/plotting_lib')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/semi_supervised_CC_GAN')

import chexpert_load
import load_model
import cc_gan

use_cuda = True
if use_cuda and torch.cuda.is_available():
    print("using CUDA")
    device = torch.device('cuda:0')
else:
    print("CUDA didn't work,using cpu")
    device = torch.device('cpu')
print(device)

def train_ccgan(method="CC_GAN",resize = 320, num_epochs=3, lr=0.0002, batch_size=16,noised=1, from_checkpoint=None, from_Pretrained = None ,root_PATH = root_PATH ,root_PATH_dataset=root_PATH_dataset, saved_model_PATH=saved_model_PATH, show=False):

    kwargs_cc_gan_patch ={"show":show, "gpu" : use_cuda}
    kwarg_Common ={"num_epochs":num_epochs,"learning_rate":lr,"batch_size":batch_size}
    kwargs={"Common":kwarg_Common}

    #just ToTensor before patch
    channel3 = transforms.Lambda(lambda x: torch.cat([x, x, x], 0))
    transform_train= transforms.Compose([ transforms.Resize((resize,resize)), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])#,transforms.Lambda(lambda x: torch.cat([x, x, x], 0))])


    #after patch transformation
    #transform_after_patching= transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    labels_path= root_PATH + "SummerThesis/code/custom_lib/chexpert_load/labels.pt"
    cheXpert_train_dataset, dataloader = chexpert_load.chexpert_load(root_PATH + "SummerThesis/code/custom_lib/chexpert_load/train.csv",
                                                                     transform_train,batch_size, labels_path=labels_path,root_dir = root_PATH_dataset, num_workers=5)

    #just to check how well G outputs ,no training in this
    _, valid_dataloader = chexpert_load.chexpert_load(root_PATH + "SummerThesis/code/custom_lib/chexpert_load/valid.csv",
                                                                                 transform_train,8, labels_path=labels_path,root_dir = root_PATH_dataset,shuffle = False)

    fixed_noise = torch.randn(batch_size,100,1, 1).to(device = device)
    iter_fixedvalid =iter(valid_dataloader)

    valid_batch = next(iter_fixedvalid)[0]
    valid_batch2 = next(iter_fixedvalid)[0]

    patcher = cc_gan.Patcher_CC_GAN(valid_batch,**kwargs_cc_gan_patch)#, transform = transform_after_patching)
    fixed_context_conditioned,fixed_low_res ,fixed_cord = patcher()

    patcher2 = cc_gan.Patcher_CC_GAN(valid_batch2,**kwargs_cc_gan_patch)#, transform = transform_after_patching)
    fixed_context_conditioned2,fixed_low_res2 ,fixed_cord2 = patcher2()



    # Plot some training images
    '''
    real_batch = next(iter(dataloader))
    real_batch2 = next(iter(dataloader))

    plt.figure(figsize=(8,4))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(torch.cat((real_batch[0].to(device)[:64],real_batch2[0].to(device)[:64])), padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    '''

    netD = cc_gan.Discriminator()
    netG = cc_gan.Generator(noise=noised,noise_k_size=int(resize/16))
    G_losses = []
    D_losses = []
    C_losses = []
    sig = nn.Sigmoid()
    advs_criterion = nn.BCELoss().to(device)
    classification_criterion = nn.BCEWithLogitsLoss(pos_weight=chexpert_load.load_posw()).to(device=device)


    beta1=0.5
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    real_label = 1
    fake_label = 0


    if from_Pretrained :
        loader = load_model.Load_Model(method=method,pre_trained = pre_trained_PATH, kwargs=kwargs, model=[netG,netD], optimizer=[optimizerG,optimizerD], plot_loss=[G_losses,D_losses] )
        file_name  ,plot_loss  = loader()

    elif from_checkpoint :
        loader = load_model.Load_Model(method=method ,from_checkpoint = from_checkpoint,  kwargs=kwargs, model=[netG,netD], optimizer=[optimizerG,optimizerD], plot_loss=[G_losses,D_losses] )
        file_name ,plot_loss  = loader()

    else:
        print("training GAN from scratch")
        file_name=method +"_"+ "".join([key + str(kwargs["Common"][key]) for key in kwargs["Common"].keys()])+".tar"


    saved_model_PATH = saved_model_PATH+  "saved_models/semi_supervised/"+ file_name[:-4]
    if not os.path.exists(saved_model_PATH): os.mkdir(saved_model_PATH)

    img_list = []
    c_loss = None

    currentDT = datetime.datetime.now()

    netD.to(device)
    netG.to(device)
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (real_images , class_labels , label_ok) in enumerate(dataloader):

            real_images = real_images.to(device=device,dtype=torch.float)
            # Real img to 3 channel for D and patcher for G
            patcher = cc_gan.Patcher_CC_GAN(real_images,**kwargs_cc_gan_patch)#, transform = transform_after_patching)
            real_images = torch.stack([channel3(r_im) for r_im in real_images ])
            class_labels = class_labels.to(device=device,dtype=torch.float)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)) + log()
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            label = torch.full((batch_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_images)
            # Calculate loss on all-real batch

            #gathering labeled ones for classification supervision
            index_list=[]
            for i_,l in enumerate(label_ok):
                if l==1: index_list.append(i_)
            #index_list = torch.tensor(index_list)
            if index_list:
                c_loss = classification_criterion(output[index_list,1:] ,class_labels[index_list,:])
                errD_real = advs_criterion( sig(output[:, 0].view(-1)), label) + c_loss
            else:
                #print(i,"noo index_list",index_list)
                errD_real = advs_criterion( sig(output[:, 0].view(-1)), label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = sig(output[:,0].view(-1)).mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors (context conditioned in our case)
            context_conditioned,low_res ,cord = patcher()
            # Generate fake image batch with G
            noise = torch.randn(batch_size,100,1, 1).to(device=device)
            fake = netG(context_conditioned,low_res,cord,noise)
            label.fill_(fake_label)

            '''PROBLEEEEEEm'''
            hole_size = cord[0]
            for idx , f_i in enumerate(fake):
                mask = torch.zeros(resize,resize).to(device=device)
                row  = cord[1][idx][0]
                col  = cord[1][idx][1]
                mask[row:row+hole_size,col:col+hole_size]=1
                #fake[idx].data = fake[idx]*mask+(1-mask)*context_conditioned[idx]
                fake[idx].data = torch.where(mask.byte(),fake[idx],context_conditioned[idx])

            fake = torch.stack([channel3(f_im) for f_im in fake ])
            # Classify all fake batch with D (din't forget to detach because you just optimize discrimantor)
            output = netD(fake.detach())
            # Calculate D's loss on the all-fake batch
            errD_fake =  advs_criterion( sig(output[:, 0].view(-1)), label)
            # Calculate the gradients for this batch
            errD_fake.backward()


            D_G_z1 = sig(output[:,0].view(-1)).mean().item()
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
            output = netD(fake)
            # Calculate G's loss based on this output
            errG = advs_criterion(sig(output[:, 0].view(-1)), label) #+ classification_criterion(output[1:], class_labels)
            # Calculate gradients for G
            errG.backward()

            D_G_z2 = sig(output[:,0].view(-1)).mean().item()
            # Update G
            optimizerG.step()

            #print("breaking");break
            # Output training stats
            if (i+1) % 100 == 0:
                 aftertDT = datetime.datetime.now()
                 c = aftertDT-currentDT
                 mins,sec=divmod(c.days * 86400 + c.seconds, 60)
                 print(mins,"mins ", sec,"secs")

                 print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch+1, num_epochs, i+1, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if (i+1) % 200 == 0:
                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if c_loss:
                    C_losses.append(c_loss.item())


            # Check how the generator is doing by saving G's output on fixed_noise
            if (i % 6000 == 0) or((epoch == num_epochs-1) and (i == len(dataloader)-1)):

                # Generate fake image batch with G
                with torch.no_grad():

                    netG.eval()
                    fake = netG(fixed_context_conditioned,fixed_low_res,fixed_cord,fixed_noise).detach().cpu()
                    fake2 = netG(fixed_context_conditioned2,fixed_low_res2,fixed_cord2,fixed_noise).detach().cpu()
                    netG.train()
                    Gen_fake = fake.detach().clone().cpu()
                    Gen_fake2 = fake2.detach().clone().cpu()

                hole_size = fixed_cord[0]
                hole_size2 = fixed_cord2[0]

                for idx , f_i in enumerate(fake):
                    mask=torch.zeros(1,resize,resize)
                    mask2=torch.zeros(1,resize,resize)

                    row = fixed_cord[1][idx][0];row2 = fixed_cord2[1][idx][0]
                    col = fixed_cord[1][idx][1];col2 = fixed_cord2[1][idx][1]
                    mask[:,row:row+hole_size,col:col+hole_size]=1
                    mask2[:,row2:row2+hole_size2,col2:col2+hole_size2]=1


                    fake[idx] = torch.where(mask.byte(),fake[idx],fixed_context_conditioned[idx].cpu())
                    fake2[idx] = torch.where(mask2.byte(),fake2[idx],fixed_context_conditioned2[idx].cpu())

#                    fake[idx].data = fake[idx]*mask+(1-mask)*context_conditioned[idx]
#                    fake2[idx].data = fake2[idx]*mask2+(1-mask2)*context_conditioned2[idx]

                img_list.append(vutils.make_grid(torch.cat((valid_batch,fixed_context_conditioned.cpu(),Gen_fake,fake,
                                                            valid_batch2,fixed_context_conditioned2.cpu(), Gen_fake2,fake2)).cpu(), padding=2, normalize=True))

    print('training done')
    aftertDT = datetime.datetime.now()
    c=aftertDT-currentDT
    mins,sec=divmod(c.days * 86400 + c.seconds, 60)
    print(mins,"mins ", sec,"secs")
    print('END--',file_name)

    PATH =  saved_model_PATH+"/"+file_name

    plt.figure(figsize=(15,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(saved_model_PATH+'/'+'plot_DnG_loss_'+ '.png')

    plt.figure(figsize=(15,5))
    plt.title("classification loss")
    plt.plot(C_losses)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.savefig(saved_model_PATH+'/'+'plot_loss_'+ '.png')

    Writer = animation.writers["ffmpeg"]
    fig, ax = plt.subplots(1,figsize=(64,64))
    plt.axis("off")
    '''
    ax[0].set_ylabel('Orginal Image');ax[0].set_ylabel('Cropped Image');ax[0].set_ylabel('Generated Image');ax[0].set_ylabel('Inpainted image')
    ax[0].set_xlabel('common xlabel')
    ax[0].set_xlabel('common xlabel')
    ax[0].set_xlabel('common xlabel')
    '''
    ims = [[plt.imshow(np.transpose(img.numpy(),(1,2,0)), animated=True)] for img in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    #ims2 = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    #ani2 = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    ani.save(saved_model_PATH+"/G_during_training.mp4")

    torch.save({
                'epoch': kwarg_Common["num_epochs"] ,
                'G_model_state_dict': netG.state_dict(),
                'G_optimizer_state_dict': optimizerG.state_dict(),
                'G_loss':G_losses,
                'D_model_state_dict': netD.state_dict(),
                'D_optimizer_state_dict': optimizerD.state_dict(),
                'D_loss':D_losses,
                'C_loss':C_losses,
                'img_lists':img_list}, PATH)

    '''
    head_name_list = [head["head_name"]  for head in head_arch]
    head_state_list = [head["head"].state_dict()  for head in head_arch]
    optimizer_state_list=[head['optimizer'].state_dict()  for head in head_arch]
    '''
    #torch.save(model.state_dict(), PATH)
    print('saved  model(model,optim,loss, epoch)')

#'''(method="relative_position",num_epochs=3, learning_rate=0.0001, batch_size=16,split = 3.0, grid_crop_size=225,patch_crop_size=64,perm_set_size=300)'''

p = saved_model_PATH +'saved_models/semi_supervised/'
schedule=[ {"num_epochs":3,"show":False, "resize":128,"batch_size":32}]


for kwargs in schedule:
  train_ccgan(**kwargs)











































#
