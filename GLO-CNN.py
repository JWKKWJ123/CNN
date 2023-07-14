import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchsummary import summary






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
        #self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.block1 = vgg_block(16, 16)
        self.block2 = vgg_block(32, 32)
        self.block3 = vgg_block(48, 48)
        self.block4 = vgg_block(64, 64)
        self.conv_cls = nn.Sequential(
            nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
            nn.Flatten(start_dim=1),
            #nn.LazyLinear(512),
            nn.Linear(80,32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32,1),
            nn.Sigmoid()
            )
        #self.block1
        #self.block1
        #self.block1


        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        x = self.conv_cls(x)
        #x = torch.sigmoid_(x)
        return x
        
    
'''
net = VGG()

X = torch.randn(size=(1,1,137,177,144), dtype=torch.float32)

Y = net(X)
print(Y.shape)

summary(net,(1,137,177,144))

'''




        
        
        
        
    
        
