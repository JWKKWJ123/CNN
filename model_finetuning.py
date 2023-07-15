from Glo-CNN import Global
from Loc-CNN import Local
import math
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from torchvision.io.image import read_image
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import einops
import glob
import time
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


#function used for setting earlyStopping for CNN
class EarlyStopping:
   
    def __init__(self, patience=40, verbose=False, delta=0, path='   ', trace_func=print):
       
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = val_loss 

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss
        
        
#function to save the curve of accuracy in training and validation set
def showlogs(logs,ki):
        logs = np.array(torch.tensor(logs, device='cpu'))
        plt.plot( )
        plt.plot(logs[:, 1])
        plt.plot(logs[:, 2])
        plt.title("model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("epoch")
        plt.legend(["train","val"],loc="lower right")
        plt.tight_layout()
        plt.savefig('_{}.png'.format(ki))
        

#function to save the loss in training and validation set
def showloss(logs,ki):
        logs = np.array(torch.tensor(logs, device='cpu'))
        plt.plot( )
        plt.plot(logs[:, 1])
        plt.title("model loss")
        plt.ylabel("Loss")
        plt.xlabel("epoch")
        plt.legend(["train","val"],loc="lower right")
        plt.tight_layout()
        plt.savefig('_{}.png'.format(ki))
        


#dataloader
class SelfDataset(Dataset):
    def __init__(self, data_dir,ki=0, K=5, typ='train'):
        self.images = []
        self.labels = []
        self.names = []
        self.subnames = []

        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images)) 
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
			#subname is the subject ID
            self.subnames += [os.path.relpath(imgs, data_dir)[2:-4] for imgs in images]
        #set split training set (90%) and testing set (10%)    
        ss1=StratifiedShuffleSplit(n_splits=2,test_size=0.1,random_state= 0)
        train_index, test_index = ss1.split(self.images, self.labels)
        test_index = train_index[1]
        train_index = train_index[0]
        self.images, X_test = np.array(self.images)[train_index], np.array(self.images)[test_index]#???????
        self.labels, y_test = np.array(self.labels)[train_index], np.array(self.labels)[test_index]#???????
        self.subnames, name_test = np.array(self.subnames)[train_index], np.array(self.subnames)[test_index]#???????
        
        
        #set K fold cross-validation in the split training set
        sfolder = StratifiedKFold(n_splits=K,random_state=0,shuffle=True)
        i=0
		#set training and validation set in each fold
        for train, test in sfolder.split(self.images,self.labels):
            if i==ki:
               if typ == 'val':
                  self.crossimages = np.array(self.images)[test]
                  self.crosslabels = np.array(self.labels)[test]
                  self.crossnames = np.array(self.subnames)[test]
               elif typ == 'train':
                  self.crossimages = np.array(self.images)[train]
                  self.crosslabels = np.array(self.labels)[train]
                  self.crossnames = np.array(self.subnames)[train]  
            i=i+1
           
        
    def __getitem__(self, idx):
        image = np.load(self.crossimages[idx])
		#image crop to (150,180,150)
        image = image[15:165,20:200,15:165]
        image = (image - np.mean(image))/ np.std(image)
        label = self.crosslabels[idx]
        name = self.crossnames[idx]
        return image, label, name
    def __getname__(self):
        return self.crossnames
    def __len__(self):
        return len(self.crossimages)
        
        
#the root of data, subjects of different categories are placed in different subfolders
root = '/...AD_CN/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#hyperparameter settings
batchs = 32
K=5
epoch = 300
learning_rate = 0.0001
#unweighted class entropy： loss_fn = nn.BCELoss()
#weighted class entropy：
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ ]))
loss_fn = loss_fn.to(device)



for ki in range(K):
    
    trainset = SelfDataset(root, ki=ki, typ='train')
    testset = SelfDataset(root, ki=ki, typ='val')
    train_data_size = len(trainset)
    test_data_size = len(testset)
    train_dataloader = DataLoader(
         dataset=trainset,
         batch_size=batchs,
         shuffle=True)
    test_dataloader = DataLoader(
         dataset=testset,
         batch_size=batchs,
     )
	#set the patience of earlyStopping and the address to save the trained model
    early_stopping = EarlyStopping(patience=40, path='_{}.pth'.format(ki))
	#load model
	model = torch.load("   ",map_location=torch.device(device) )
    model = model.to(device)
	#set optimizer
    optimzer = torch.optim.Adam(params= model.parameters(), lr = learning_rate)
    # num of training steps
    total_train_step = 0
    # num of testing steps
    total_test_step = 0
    logs = []
    loss_list = []
    for i in range(epoch):
      print("--------start training epoch: {}-------------------".format(i + 1))
      # start of training
      model.train()
      total_test_loss = 0
      total_trainaccuracy = 0
      total_testaccuracy = 0
      for data in train_dataloader:
        imgs, targets ,names = data
        imgs = imgs.unsqueeze(1)
        imgs = imgs.to(torch.float32)
        targets = targets.unsqueeze(1)
        targets = targets.to(torch.float32)
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        accuracy = (torch.where(outputs > 0.5, 1, 0) == targets).sum()
        total_trainaccuracy = total_trainaccuracy + accuracy
        predicts = torch.where(outputs > 0.5, 1, 0)
        loss = loss_fn(outputs, targets)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("num of training:{},loss:{}".format(total_train_step, loss.item()))
    # start of testing
      model.eval()
      with torch.no_grad():
        for data in test_dataloader:
            imgs, targets,names = data
            imgs = imgs.unsqueeze(1)
            imgs = imgs.to(torch.float32)
            targets = targets.unsqueeze(1)
            targets = targets.to(torch.float32)
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (torch.where(outputs > 0.5, 1, 0) == targets).sum()
            total_testaccuracy = total_testaccuracy + accuracy
      #if the loss in validation set do not decrease, then stop training
      early_stopping(total_test_loss, model)
      if early_stopping.early_stop:
         print("Early stopping")
         break
      print("train accuracy: {}".format(total_trainaccuracy / train_data_size))
      print("test Loss: {}".format(total_test_loss))
      print("test accuracy: {}".format(total_testaccuracy / test_data_size))
      logs.append([epoch, total_trainaccuracy/train_data_size, total_testaccuracy/test_data_size])
      loss_list.append([epoch,total_test_loss])
      total_test_step += 1
	  #save accuracy curve and loss curve
      showloss(loss_list,ki)
      plt.close()
      showlogs(logs,ki)
      plt.close()  
        
        
        
