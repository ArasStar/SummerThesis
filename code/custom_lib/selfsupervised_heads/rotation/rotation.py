import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches4rectangle


class Rotation(object):
    def __init__(self, image_batch, K=4, gpu=False, show=False,transform= None, resize = 320):

        self.bs, _, self.h, self.w = image_batch.shape
        self.show = show  # print cropped images and show where they are 1 time for each batch
        self.image_batch = image_batch
        self.K = K
        self.device = torch.device('cuda') if gpu else torch.device('cpu')
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((resize,resize))]) if resize!=320  else transform

    def __call__(self):

        patches = torch.Tensor()
        labels =  torch.Tensor()

        for idx, image in enumerate(self.image_batch):
            #print(image.shape)
            rotated_patches ,label  = self.rotated_patches(image)

            if idx == 1 and self.show:
                self.show_rotated_patches(rotated_patches, label)

            if self.transform:
                 rotated_patches = torch.stack([self.transform(x_i)
                     for i, x_i in enumerate(torch.unbind(rotated_patches, dim=0))], dim=0)

            patches = torch.cat((patches, rotated_patches))
            labels = torch.cat((labels, label))

        return patches, labels

    def rotated_patches(self,image):
        rotated_patches = torch.empty(self.K,3,self.h, self.w)
        labels = torch.arange(self.K).type(torch.FloatTensor)


        if self.K==4:
         #0 degree
         rotated_patches[0]= image
         # 90 degree
         rotated_patches[1]=image.permute([0,2,1]).flip(dims=[1])

         #180 degree
         rotated_patches[2]=image.flip(dims=[1]).flip(dims=[2])

         #270 degree
         rotated_patches[3]=image.flip(dims=[1]).permute([0,2,1])

        else:
            print("not implemented other K values then 4")

        return rotated_patches,labels

    def show_rotated_patches(self,rotated_images,labels):

        fig, ax = plt.subplots(1, self.K, sharex='col', sharey='row', figsize=(10, 3))

        for k in range(self.K):
            r = transforms.ToPILImage()(rotated_images[k,0, :, :])
            ax[k].imshow(r, cmap='Greys_r')
            ax[ k].set_title(str(k*90)+ " degree")

        plt.show()


class Basic_Head(torch.nn.Module):
    def __init__(self, D_in, D_out=4, gpu=True):
        """
        No task head just concating what comes out just before default classifer then applying linear
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Basic_Head, self).__init__()
        self.device = torch.device('cuda:0') if gpu else torch.device('cpu')
        self.classifier = torch.nn.Linear(D_in,D_out).to(device = self.device)

    def forward(self, x):
        #linear output with 8 outpts(directions)

        return self.classifier(x)
