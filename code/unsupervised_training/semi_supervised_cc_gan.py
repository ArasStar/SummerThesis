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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CheXpertDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):

        super().__init__()

        self.observations_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.list_classes=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    def __len__(self):
        return len(self.observations_frame)

    def __getitem__(self, idx):

        img_name = os.path.join(self.observations_frame.iloc[idx, 0])
        image = imread(img_name)
        image = transforms.ToPILImage()(image)

        observations=self.observations_frame.loc[idx,self.list_classes]
        observations = torch.from_numpy(observations.values.astype(np.float32))


        #RETURNING IMAGE AND LABEL SEPERATELY FOR TORCH TRANSFORM LIB

        if self.transform:

            image = self.transform(image)

        return image,observations

def chexpert_load(csv_file_name,transformation,batch_size):

  cheXpert_dataset = CheXpertDataset(csv_file=csv_file_name,
                                     root_dir='not used', transform=transformation)

  dataloader = DataLoader(cheXpert_dataset, batch_size=batch_size,shuffle=True)

  return cheXpert_dataset,dataloader



use_cuda = True
if use_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("CUDA didn't work")
    device = torch.device('cpu')

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.patches as patches4rectangle
import torch


fig, ax = plt.subplots(1)
im = Image.open("BU.jpg")
draw = ImageDraw.Draw(im)


"""


fontsize = 35
font = ImageFont.truetype("arial.ttf", fontsize)

draw.line((0, 75, im.size[0], 75), fill=8,width=5)
draw.line((0, 150, im.size[0], 150), fill=8,width=5)
draw.line((75, 0, 75, im.size[1]), fill=8,width=5)
draw.line((150, 0, 150, im.size[1]), fill=8,width=5)

perm = [0,1,2,3,4,5,6,7,8]
perm=iter(perm)
i=0
for row in range(0,3):
  for col in range(0,3):
    draw.text((0 + col*75, 0 + row*75),str(next(perm)), fill="#ff0000", font = font)
    num= num + 1

del draw
"""

start_col = 30
start_row=100

for i in range(0,4):
  shift= i*75
  ax.plot([start_col,start_col+75*3],[start_row+shift,start_row+shift],color='r',linestyle='dashed')
  ax.plot([start_col+shift,start_col+shift],[start_row,start_row+75*3],color='r',linestyle='dashed')


rect = patches4rectangle.Rectangle((0, 0), 75 , 75, linewidth=2, edgecolor='r', facecolor='none',linestyle="--")
#ax.add_patch(rect)
#rect = patches4rectangle.Rectangle((80, 0), 75 , 75, linewidth=2, edgecolor='r', facecolor='none')
#ax.add_patch(rect)
plt.imshow(im, cmap='Greys_r')

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.patches as patches4rectangle
import matplotlib.gridspec as gridspec
import torch



fig, ax = plt.subplots(3,3, sharex='col', sharey='row')
fig.set_figheight(8.2)
fig.set_figwidth(12)
fig.subplots_adjust(hspace=0,wspace=0)

#gs1 = gridspec.GridSpec(3, 3)
#gs1.update(wspace=0.025, hspace=0.05)

# row and column sharing
im = Image.open("BU.jpg")
draw = ImageDraw.Draw(im)

start_col = 30
start_row=100
"""
for i in range(0,4):
  shift= i*75
  ax.plot([start_col,start_col+75*3],[start_row+shift,start_row+shift],color='r',linestyle='dashed')
  ax.plot([start_col+shift,start_col+shift],[start_row,start_row+75*3],color='r',linestyle='dashed')

"""
rect = patches4rectangle.Rectangle((0, 0), 75 , 75, linewidth=2, edgecolor='r', facecolor='none', linestyle="--")




for i in range(3):
  for j in range(3):

    ax[i,j].imshow(im, cmap='Greys_r')
    ax[i,j].axis('off')

from PIL import Image, ImageDraw, ImageFont

image = Image.open("BU.jpg")
draw = ImageDraw.Draw(image)

txt = "Hello World"
fontsize = 1  # starting font size

# portion of image width you want text width to be
img_fraction = 0.50

font = ImageFont.truetype("arial.ttf", fontsize)
while font.getsize(txt)[0] < img_fraction*image.size[0]:
    # iterate until the text size is just larger than the criteria
    fontsize += 1
    font = ImageFont.truetype("arial.ttf", fontsize)

# optionally de-increment to be sure it is less than criteria
fontsize -= 1
font = ImageFont.truetype("arial.ttf", fontsize)

print('final font size',fontsize)
draw.text((10, 25), txt, font=font) # put the text on the image
plt.imshow(image, cmap='Greys_r')

image.save('hsvwheel_txt.png') # save it

x = np.arange(9)
np.random.shuffle(x)
print(x)

result = np.where(x == 5)[0][0]

print(type(result))

2<=2
