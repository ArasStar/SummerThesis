import torch
import torch.nn as nn
import torch.nn.functional as F
import modified_densenet
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

import modified_densenet

class Generator(torch.nn.Module):
    #generator inspire by DCGAN generator
    def __init__(self, D_in=1, noise=0,noise_k_size=8, noise_size=100):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Generator, self).__init__()
        self.noise = noise
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


        ngf=64#3-64-128-256-512-256-128-64-3
        self.conv_layer1 = nn.Conv2d(D_in, ngf, kernel_size=4, stride=2, padding=1)
        self.bn1 =  nn.BatchNorm2d(ngf)

        self.conv_layer2 = nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d( ngf*2)

        self.conv_layer3 = nn.Conv2d( ngf*2+1, ngf*4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d( ngf*4)

        self.conv_layer4 = nn.Conv2d( ngf*4, ngf*8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d( ngf*8)


        self.noise_upconv = nn.ConvTranspose2d( noise_size, ngf*8, kernel_size=noise_k_size)
        self.bn_noise = nn.BatchNorm2d( ngf*8)

        self.upconv_layer5 = nn.ConvTranspose2d( (1+self.noise)*ngf*8,  ngf*4, kernel_size=4, stride=2,padding=1)
        self.bn5 = nn.BatchNorm2d( ngf*4)

        self.upconv_layer6 = nn.ConvTranspose2d( ngf*4,  ngf*2, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d( ngf*2)

        self.upconv_layer7 = nn.ConvTranspose2d( ngf*2, ngf, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(64)

        self.upconv_layer8 = nn.ConvTranspose2d(ngf, D_in, kernel_size=4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(D_in)

        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))


    def forward(self,context_x, lowres_x, cord, noise=0):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.relu(self.bn1(self.conv_layer1(context_x)))
        x = self.relu(self.bn2(self.conv_layer2(x)))

        #putting the conditioning
        assert(x.shape[2]==lowres_x.shape[2]), "dimensions not matching between conv img and lowres img "+str(x.shape)+"--"+str(lowres_x.shape)

        x = torch.stack([torch.cat((x_i,lowres_x[i]))
            for i, x_i in enumerate(torch.unbind(x, dim=0))], dim=0)

        x = self.relu(self.bn3(self.conv_layer3(x)))
        x = self.relu(self.bn4(self.conv_layer4(x)))

        if isinstance(noise,torch.Tensor):
            assert(self.noise == 1), "you have to init the gen() with noise =1 "
            noise_x = self.relu(self.bn_noise(self.noise_upconv(noise)))
            x = torch.stack([torch.cat((x_i,noise_x[i]))
                for i, x_i in enumerate(torch.unbind(x, dim=0))], dim=0)

        x = self.relu(self.bn5(self.upconv_layer5(x)))
        x = self.relu(self.bn6(self.upconv_layer6(x)))
        x = self.relu(self.bn7(self.upconv_layer7(x)))
        x = self.tanh(self.bn8(self.upconv_layer8(x)))

        #print("x.hsape",x.shape) #16:3: 320 320
        #print("context_x", context_x.shape)#16:3: 320 320

        return x


class Discriminator(torch.nn.Module):
    #discriminator its basically a deep model(Densnet) relu s changed to leaky relu
    def __init__(self,pretrained=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Discriminator, self).__init__()

        #Densenet with relu s replaced with leaky relu
        self.discriminator = modified_densenet.densenet121(num_classes=6, pretrained=pretrained)
        if not pretrained :
            for m in self.discriminator.modules():
                if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.Linear)):
                    nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))

    def __call__(self,x):
         return self.discriminator(x)



class Patcher_CC_GAN(object):
    def __init__(self, image_batch,hole_size="",transform ="",show=False, gpu = True):

        self.bs, _, self.h, _ = image_batch.shape
        self.image_batch = image_batch
        self.device = torch.device('cuda:0') if gpu else torch.device('cpu')
        self.show = show  # print cropped images and show where they are 1 time for each batch

        self.hole_size = int(self.h / 2) if hole_size=="" else hole_size
        self.transform = transform


    def __call__(self):
        cropped_images = torch.Tensor().to(self.device)
        low_res_images= F.interpolate(self.image_batch[:,0].view(self.bs,1,self.h,self.h), scale_factor= 1.0/4, mode='bilinear').view(self.bs,1,int(self.h/4),int(self.h/4))
        shift_row_col=[]

        for idx, image in enumerate(self.image_batch):
            cropped_image , crop ,shiftrow ,shiftcol = self.random_hole(image)
            shift_row_col.append([shiftrow,shiftcol])

            if idx == 1 and self.show:
                self.show_context_image(image,cropped_image,crop,low_res_images[idx])

            if self.transform:
                 cropped_image = self.transform(cropped_image)
                 #low_res_images = self.transform(low_res_images)

            cropped_images = torch.cat((cropped_images, cropped_image.view(1,1,self.h,self.h).to(device=self.device)))

        return cropped_images, low_res_images.to(device=self.device)  ,[self.hole_size, shift_row_col]   #,crops


    def random_hole(self, image):
        [shiftrow, shiftcol] = np.random.randint(self.h - self.hole_size - 1, size=2)

        crop = image[:, shiftrow:shiftrow + self.hole_size, shiftcol:shiftcol + self.hole_size].clone()

        cropped_image = image.clone()
        cropped_image[:,  shiftrow:shiftrow + self.hole_size, shiftcol:shiftcol + self.hole_size] = torch.zeros(self.hole_size,self.hole_size)

        return cropped_image , crop ,shiftrow ,shiftcol




    def show_context_image(self, image, cropped_image , crop ,low_res_image):
        # Preparing

        image_draw = transforms.ToPILImage()(image)
        cropped_image_draw = transforms.ToPILImage()(cropped_image)
        crop_draw = transforms.ToPILImage()(crop)
        low_res_image_draw = transforms.ToPILImage()(low_res_image)


        # Original Image plot
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))

        #fig.axis("off")

        fig.suptitle("image - context image - cropped_image")
        ax[0].imshow(image_draw, cmap='Greys_r')
        ax[0].set_title("image "+ str(image.shape))

        ax[1].imshow(cropped_image_draw, cmap='Greys_r')
        ax[1].set_title("cropped image "+ str(cropped_image.shape))

        ax[2].imshow(crop_draw, cmap='Greys_r')
        ax[2].set_title("crop " + str(crop.shape))

        ax[3].imshow(low_res_image_draw, cmap='Greys_r')
        ax[3].set_title("low_res_image "+ str(low_res_image.shape))


        plt.show()









def show(img):

    npimg = img.numpy()
    fig, ax = plt.subplots(8,8,figsize=(64,64))

    for i in range(8):
        for j in range(8):
            pass


    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.axis("off")

    fig2, ax2 = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(5, 5))
    fig2.subplots_adjust(hspace=0, wspace=0)











#-----------------------------------------
