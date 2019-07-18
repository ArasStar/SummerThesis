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
    def __init__(self, D_in=3,):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Generator, self).__init__()

        self.conv_layer1 = nn.Conv2d(D_in, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 =  nn.BatchNorm2d(64)

        self.conv_layer2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv_layer3 = nn.Conv2d(128+1,256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv_layer4 = nn.Conv2d(256,512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.upconv_layer5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2,padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.upconv_layer6 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.upconv_layer7 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(64)

        self.upconv_layer8 = nn.ConvTranspose2d(64, D_in, kernel_size=4, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(D_in)


    def forward(self, context_x, lowres_x, cord):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = F.relu(self.bn1(self.conv_layer1(context_x)))
        x = F.relu(self.bn2(self.conv_layer2(x)))

        #putting the conditioning
        assert(x.shape[2]==lowres_x.shape[2]), "dimensions not matching between conv img and lowres img "+str(x.shape)+"--"+str(lowres_x.shape)

        x = torch.stack([torch.cat((x_i,lowres_x[i]))
            for i, x_i in enumerate(torch.unbind(x, dim=0))], dim=0)

        x = F.relu(self.bn3(self.conv_layer3(x)))
        x = F.relu(self.bn4(self.conv_layer4(x)))
        x = F.relu(self.bn5(self.upconv_layer5(x)))
        x = F.relu(self.bn6(self.upconv_layer6(x)))
        x = F.relu(self.bn7(self.upconv_layer7(x)))
        x = torch.tanh(self.bn8(self.upconv_layer8(x)))


        return context_x


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
        cropped_images = torch.Tensor()
        low_res_images=F.interpolate(self.image_batch[:,0].view(self.bs,1,self.h,self.h), scale_factor= 1.0/4, mode='bilinear').view(self.bs,1,int(self.h/4),int(self.h/4))
        shift_row_col=[]

        for idx, image in enumerate(self.image_batch):
            cropped_image , crop ,shiftrow ,shiftcol = self.random_hole( image, self.hole_size)
            shift_row_col.append([shiftrow,shiftcol])

            if idx == 1 and self.show:
                self.show_context_image(image,cropped_image,crop,low_res_image)

            if self.transform:
                 cropped_image = self.transform(cropped_image)
                 #low_res_images = self.transform(low_res_images)

            cropped_images = torch.cat((cropped_images, cropped_image.view(1,3,self.h,self.h)))

        return cropped_images , low_res_images ,[self.hole_size, shift_row_col]   #,crops


    def random_hole(self, image, hole_size):
        [shiftrow, shiftcol] = np.random.randint(self.h - hole_size - 1, size=2)

        crop = image[:, shiftrow:shiftrow + hole_size, shiftcol:shiftcol + hole_size]

        image[:,  shiftrow:shiftrow + hole_size, shiftcol:shiftcol + hole_size] = torch.zeros(hole_size,hole_size)

        return image , crop ,shiftrow ,shiftcol




    def show_cropped_patches(self, image, cropped_image , crop ,low_res_image):
        # Preparing

        image_draw = transforms.ToPILImage()(image)
        cropped_image = transforms.ToPILImage()(cropped_image)
        crop = transforms.ToPILImage()(crop)
        low_res_image = transforms.ToPILImage()(low_res_image)


        # Original Image plot
        fig, ax = plt.subplots(1, 4, figsize=(5, 5))
        ax.axis("off")

        fig.suptitle("image - context image - cropped_image")
        ax[0].imshow(image, cmap='Greys_r')
        ax[1].imshow(cropped_image, cmap='Greys_r')
        ax[2].imshow(crop, cmap='Greys_r')
        ax[3].imshow(low_res_image, cmap='Greys_r')


        plt.imshow(image_draw, cmap='Greys_r')
        plt.show()






















#-----------------------------------------
