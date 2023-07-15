from Glo-CNN import Global
from Loc-CNN import Local
import math
import pandas as pd
import numpy as np
import os
import scipy.stats as st
from scipy.special import logit, expit
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import matplotlib.pyplot as plt
from torchvision.io.image import read_image
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize, to_pil_image
import einops
import glob
import time
from sklearn.metrics import accuracy_score,recall_score,roc_curve, classification_report,confusion_matrix,precision_score,roc_auc_score, auc
import random
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import resample




#function to set the bootstrap and calculate confidence interval
n_iterations = 100
def bootstrap(a, b, calculate_statistic,name):
    # run bootstrap
    stats = list()
    for i in range(n_iterations):
        # prepare sample
        sample_a, sample_b = resample(a, b, stratify=a,  random_state=i)
        
        stat = calculate_statistic(sample_a, sample_b)
        stats.append(stat)
    average = np.average(stats)
    print(len(a),len(sample_a),str(calculate_statistic))
    metric = pd.DataFrame(stats)    
    metric.to_csv("results/{}_DL.csv".format(name))
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    return [average,lower, upper]



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
            #print(i)
            self.labels += ([i] * len(images))  # ?i????????,????i
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
            self.subnames += [os.path.relpath(imgs, data_dir)[2:-4] for imgs in images]
            
            
        ss1=StratifiedShuffleSplit(n_splits=2,test_size=0.1,random_state= 0)
        
        train_index, test_index = ss1.split(self.images, self.labels)
        test_index = train_index[1]
        train_index = train_index[0]

        self.images, X_test = np.array(self.images)[train_index], np.array(self.images)[test_index]#???????
        self.labels, y_test = np.array(self.labels)[train_index], np.array(self.labels)[test_index]#???????
        self.subnames, name_test = np.array(self.subnames)[train_index], np.array(self.subnames)[test_index]#???????
        self.crossimages = X_test
        self.crosslabels = y_test
        self.crossnames = name_test
        
        
    def __getitem__(self, idx):
        image = np.load(self.crossimages[idx])
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
root = '    '
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#loss_fn = nn.BCELoss()
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([  ]))
loss_fn = loss_fn.to(device)

fpr_list = []
tpr_list = []
acc_list = []
sen_list = []
spe_list = []
auc_list = []
biomarker_list = []




trainset = SelfDataset(root, K=K, typ='train')
testset = SelfDataset(root, K=K, typ='val')
train_data_size = len(trainset)
test_data_size = len(testset)
    
print(train_data_size,test_data_size)
    
train_dataloader = DataLoader(
         dataset=trainset,
         batch_size=batchs,
         shuffle=True)
test_dataloader = DataLoader(
         dataset=testset,
         batch_size=batchs,
     )
     
    # num of training steps
total_train_step = 0
    # num of testing steps
total_test_step = 0
logs = []
#load model
model = torch.load("   ",map_location=torch.device(device) )
model.eval()
output = []
label = []
output = torch.empty(0,1).to(device)
label = torch.empty(0,1).to(device)
i=0
with torch.no_grad():
 for data in test_dataloader:
            imgs, targets, names = data
            imgs = imgs.unsqueeze(1)
            imgs = imgs.to(torch.float32)
            print(targets)
            targets = targets.unsqueeze(1)
            targets = targets.to(torch.float32)
            imgs = imgs.to(device)
            targets = targets.to(device)
            names = np.array(names).reshape(-1,1)
            outputs = model(imgs)
            output_biomarker = outputs.cpu().detach().numpy()
            targets_output = targets.cpu().detach().numpy()
            biomarkers = np.concatenate((names,targets_output,output_biomarker),axis=1)
            biomarkers = np.squeeze(biomarkers,axis=0)
            biomarker_list.append(biomarkers)
            output = torch.cat([output,outputs],0)
            label = torch.cat([label,targets],0)
            accuracy = (torch.where(outputs > 0.5, 1, 0) == targets).sum()

output = output.cpu().numpy()
label = label.cpu().numpy()
output_pro = output
output = np.where(output > 0.5, 1, 0)
#calculate metrics
auc=roc_auc_score( label,output_pro)
auc_list.append(auc)
fpr, tpr, thresholds = roc_curve(label,output_pro)
fpr_list.append(fpr)
tpr_list.append(tpr)
report=classification_report(label,output)
accuracy=accuracy_score(label,output)
recall=recall_score(label,output)
precision=precision_score(label,output)
matrix=confusion_matrix(label,output)
sensitivity = float(matrix[1][1])/np.sum(matrix[1])
specificity = float(matrix[0][0])/np.sum(matrix[0][0]+matrix[0][1])
acc_list.append(accuracy)
sen_list.append(sensitivity)
spe_list.append(specificity)
print('accuracy:'+str(accuracy))
print('recall:'+str(recall))
print('precision:'+str(precision))
print('matrix:'+str(matrix))
print('report:'+str(report))
print('AUC:'+str(auc))
print('sensitivity:',sensitivity)
print('specificity:',specificity)
with open('   .txt','w') as f:
      f.write(report)
      f.write(str(accuracy)+ os.linesep)
      f.write(str(sensitivity)+'\n')
      f.write(str(specificity)+'\n')
      f.write(str(auc)+'\n')
      


#do bootstrap in testing set： 
ci_acc = bootstrap(label,output , accuracy_score,'CNN_acc_ADNI') 
ci_auc = bootstrap(label,output_pro , roc_auc_score,'CNN_auc_ADNI') 

print(ci_acc)
print(ci_auc)
with open('   .txt','w') as f:
      f.write(str(ci_acc)+'\n')
      f.write(str(ci_auc)+'\n')
 

 
#output the DL-biomarkers for the subjects in testing set           
biomarker_list = pd.DataFrame(data=biomarker_list)
biomarker_list.to_csv('  .csv',index_label='index')



      

                                                  

















































