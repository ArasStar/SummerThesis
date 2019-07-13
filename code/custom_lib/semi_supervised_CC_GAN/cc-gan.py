import torch.nn as nn
import torch.nn.functional as F



class Generator(torch.nn.Module):
    #generator inspire by DCGAN generator
    def __init__(self, D_in=1, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Gen, self).__init__()

        self.conv_layer1 = nn.Conv2d(D_in, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 =  nn.BatchNorm2d(64)

        self.conv_layer2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv_layer3 = nn.Conv2d(128+1,256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv_layer4 = nn.Conv2d(256,512 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.upconv_layer5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2)
        self.bn5 = nn.BatchNorm2d(256)

        self.upconv_layer6 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2)
        self.bn6 = nn.BatchNorm2d(128)

        self.upconv_layer7 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.bn7 = nn.BatchNorm2d(64)

        self.upconv_layer8 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2)
        self.bn8 = nn.BatchNorm2d(3)


    def forward(self, context_x,lowres_x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = F.relu(self.bn1(self.conv_layer1(context_x)))
        x = F.relu(self.bn2(self.conv_layer2(x)))

        #putting the conditioning
        assert(x.shape[1]==lowres_x.shape[1]) "dimensions not matching between conv img and lowres img "
        x=torch.cat((x,lowres_x))

        x = F.relu(self.bn3(self.conv_layer3(x)))
        x = F.relu(self.bn4(self.conv_layer4(x)))
        x = F.relu(self.bn5(self.upconv_layer5(x)))
        x = F.relu(self.bn6(self.upconv_layer6(x)))
        x = F.relu(self.bn7(self.upconv_layer7(x)))
        x = F.Tanh(self.bn8(self.upconv_layer8(x)))

        return x


class Discriminator(torch.nn.Module):
    #generator inspire by DCGAN generator
    def __init__(self, D_in=1, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Gen, self).__init__()

        self.conv_layer1 = nn.Conv2d(D_in, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 =  nn.BatchNorm2d(64)

        self.conv_layer2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv_layer3 = nn.Conv2d(128+1,256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv_layer4 = nn.Conv2d(256,512 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.upconv_layer5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2)
        self.bn5 = nn.BatchNorm2d(256)

        self.upconv_layer6 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2)
        self.bn6 = nn.BatchNorm2d(128)

        self.upconv_layer7 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.bn7 = nn.BatchNorm2d(64)

        self.upconv_layer8 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2)
        self.bn8 = nn.BatchNorm2d(3)
