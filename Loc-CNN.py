import torch
import torch.nn as nn
from torchsummary import summary
import copy
import math


#3 convolution layers
class vgg_block(nn.Module):
    def __init__(self, inplanes, planes):
        super(vgg_block, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1,bias=False)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1,bias=False)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1,bias=False)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)
        self.bn3 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        return out


        
class VGG(nn.Module):
    def __init__(self):
        super(dyrbaVGG, self).__init__()
		#The input and output of vgg_block( ) is the number of channels
		#1 convolution block
        self.block1 = vgg_block(1, 10)
		#output block, contain 3 FC layers
        self.conv_cls = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64,1),
            nn.Sigmoid()
            )

        
    def forward(self, x):
        x = self.block1(x)
        x = self.conv_cls(x)
        return x


class Local(nn.Module):
    def __init__(self,inplace,
                 patch_size=30,
                 backbone=VGG,
				 LOC = [0,0,0]):
        """
        Parameter:
            @patch_size: the patch size of the local pathway
            @backbone: the backbone of extract the features
			@LOC: the location of the input patch in the brain image
        """
        
        super().__init__()
        
        self.patch_size = patch_size
        self.step = step
        self.hidden_size = 10
        self.LOC = LOC
        self.cnn = backbone()
        
        
        
    def forward(self,xinput):
        B,C,W,D,H=xinput.size()
        x = self.LOC[0]
        z = self.LOC[1]
        y = self.LOC[2]
        locx = xinput[:,:,x:x+self.patch_size,z:z+self.patch_size,y:y+self.patch_size]
        xloc = self.cnn(locx)              
      
        return xloc 
                
                
    
