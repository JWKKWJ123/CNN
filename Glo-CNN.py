import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchsummary import summary

#convolution block (3 convolution layers) 
class vgg_block(nn.Module):
    def __init__(self, inplanes, planes):
        super(vgg_block, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(inplanes, 16 , kernel_size=3, stride=(2, 2, 2), padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes+16)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        residual = self.conv3(residual)

        out = torch.cat((out,residual),axis=1)
        out = self.bn3(out)
        out = self.relu(out)
        return out


        
class Global(nn.Module):
    def __init__(self):
        super(Global, self).__init__()

        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
		#The input and output of vgg_block( ) is the number of channels
		#4 convolution blocks
        self.block1 = vgg_block(16, 16)
        self.block2 = vgg_block(32, 32)
        self.block3 = vgg_block(48, 48)
        self.block4 = vgg_block(64, 64)
		#FC layers
        self.conv_cls = nn.Sequential(
            nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(80,32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32,1),
            nn.Sigmoid()
            )

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_cls(x)
		#output x is the possibility to be positive [0,1]
        return x
        

        
        
        
        
    
        
