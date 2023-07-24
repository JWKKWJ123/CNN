from Glo-CNN import Global
from Loc-CNN import Local
from glo import Global
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
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import einops
import glob
import time
from sklearn.metrics import accuracy_score,recall_score,roc_curve, classification_report,confusion_matrix,precision_score,roc_auc_score, auc
import random
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit


#function for add figures in the plot
def autolabel(rects, xpos='center',decimals=2):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        float_format = '{:.' + str(decimals) + 'f}'
        ax.annotate(float_format.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

					
#function to calculate confidence intervals(1)
def compute_confidence(metric, N_train, N_test, alpha=0.95):
    """
    Function to calculate the adjusted confidence interval
    metric: numpy array containing the result for a metric for the different cross validations
    (e.g. If 20 cross-validations are performed it is a list of length 20 with the calculated accuracy for
    each cross validation)
    N_train: Integer, number of training samples
    N_test: Integer, number of test_samples
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 95%
    """

    # Convert to floats, as python 2 rounds the divisions if we have integers
    N_train = float(N_train)
    N_test = float(N_test)
    N_iterations = float(len(metric))

    if N_iterations == 1.0:
        print('[WORC Warning] Cannot compute a confidence interval for a single iteration.')
        print('[WORC Warning] CI will be set to value of single iteration.')
        metric_average = np.mean(metric)
        CI = (metric_average, metric_average)
    else:
        metric_average = np.mean(metric)
        S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric)**2.0)

        metric_std = np.sqrt((1.0/N_iterations + N_test/N_train)*S_uj)

        CI = st.t.interval(alpha, N_iterations-1, loc=metric_average, scale=metric_std)

    if np.isnan(CI[0]) and np.isnan(CI[1]):
        # When we cannot compute a CI, just give the averages
        CI = (metric_average, metric_average)
    return CI

#function to calculate confidence intervals(2)
def compute_confidence_logit(metric, N_train, N_test, alpha=0.95):
    """
    Function to calculate the adjusted confidence interval
    metric: numpy array containing the result for a metric for the different cross validations
    (e.g. If 20 cross-validations are performed it is a list of length 20 with the calculated accuracy for
    each cross validation)
    N_train: Integer, number of training samples
    N_test: Integer, number of test_samples
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 95%
    """
    N_iterations = len(metric)

    # Compute average of logit function
    # metric_logit = [logit(x) for x in metric]
    logit_average = logit(np.mean(metric))

    # Compute metric average and corrected resampled t-test metric std
    metric_average = np.mean(metric)
    S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric)**2.0)
    metric_std = np.sqrt((1.0/N_iterations + N_test/N_train)*S_uj)

    # Compute z(1-alpha/2) quantile
    q1 = 1.0-(1-alpha)/2
    z = st.t.ppf(q1, N_iterations - 1)

    # Compute logit confidence intervals according to Barbiero
    theta_L = logit_average - z * metric_std/(metric_average*(1 - metric_average))
    theta_U = logit_average + z * metric_std/(metric_average*(1 - metric_average))

    # Transform back
    CI = (expit(theta_L), expit(theta_U))

    return CI



#dataloader
class SelfDataset(Dataset):
    def __init__(self, data_dir,ki=0, K=10, typ='train'):
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
            
        #select split training set and testing set for each iteration 
        ss=StratifiedShuffleSplit(n_splits=2,test_size=0.1,random_state=ki)
        train_index, test_index = ss.split(self.images,self.labels)
        
        test_index = train_index[1]
        train_index = train_index[0]

        X_trainimage, self.crossimages = np.array(self.images)[train_index], np.array(self.images)[test_index]
        X_trainlabel, self.crosslabels = np.array(self.labels)[train_index], np.array(self.labels)[test_index]
        X_trainname, self.crossnames = np.array(self.subnames)[train_index], np.array(self.subnames)[test_index]
             
        
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



root = '   '
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batchs = 32
K=10

fpr_list = []
tpr_list = []
acc_list = []
sen_list = []
spe_list = []
auc_list = []
biomarker_list = []



for ki in range(K):
    print(ki)
    trainset = SelfDataset(root, ki=ki, K=K, typ='train')
    testset = SelfDataset(root, ki=ki, K=K, typ='val')
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
     
    # num of training steps
    total_train_step = 0
    # num of testing steps
    total_test_step = 0
    logs = []
	#load model in trained in different iterations
    model = torch.load("  _{}.pth".format(ki),map_location=torch.device(device) )
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
            biomarkers = np.concatenate((names,output_biomarker),axis=1)
            biomarkers = np.squeeze(biomarkers,axis=0)
            biomarker_list.append(biomarkers)
            output = torch.cat([output,outputs],0)
            label = torch.cat([label,targets],0)
            accuracy = (torch.where(outputs > 0.5, 1, 0) == targets).sum()
    
            
    output = output.cpu().numpy()
    label = label.cpu().numpy()
    output_pro = output


    output = np.where(output > 0.5, 1, 0)
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
    print('accuracy:'+str(accuracy))
    print('recall:'+str(recall))
    print('precision:'+str(precision))
    print('matrix:'+str(matrix))
    print('report:'+str(report))
    print('AUC:'+str(auc))

    sensitivity = float(matrix[1][1])/np.sum(matrix[1])
    specificity = float(matrix[0][0])/np.sum(matrix[0][0]+matrix[0][1])
    acc_list.append(accuracy)
    sen_list.append(sensitivity)
    spe_list.append(specificity)
    print('sensitivity:',sensitivity)
    print('specificity:',specificity)
	#output 4 metrics for each iteration
    with open('result_{}.txt'.format(ki),'w') as f:
      f.write(report)
      f.write(str(accuracy)+ os.linesep)
      f.write(str(sensitivity)+'\n')
      f.write(str(specificity)+'\n')
      f.write(str(auc)+'\n')

auc_mean = np.mean(auc_list)
auc_std = np.std(auc_list)
acc_mean = np.mean(acc_list)
acc_std = np.std(acc_list)
sen_mean = np.mean(sen_list)
sen_std = np.std(sen_list)
spe_mean = np.mean(spe_list)
spe_std = np.std(spe_list)


                        
#output the average performance of 10 iterations                 
with open('   .txt','w') as f:
   f.write(str(acc_mean)+' $\pm$ '+ str(acc_std*100)+os.linesep)
   f.write(str(sen_mean)+'$\pm$'+ str(sen_std*100)+os.linesep)
   f.write(str(spe_mean)+'$\pm$'+ str(spe_std*100)+os.linesep)
   f.write(str(auc_mean)+'$\pm$'+ str(auc_std*100)+os.linesep)
    


	

###draw IC plot
#calculate the confidence interval of each metric
CI_acc = compute_confidence(acc_list, train_data_size,test_data_size)
CI_auc = compute_confidence(auc_list, train_data_size,test_data_size)
CI_sen = compute_confidence(sen_list, train_data_size,test_data_size)
CI_spe = compute_confidence(spe_list, train_data_size,test_data_size)

ind = np.arange(4)/2
width = 0.35

cnn_acc_err = (CI_acc[1]-CI_acc[0])/2
cnn_sen_err = (CI_sen[1]-CI_sen[0])/2
cnn_spe_err = (CI_spe[1]-CI_spe[0])/2
cnn_auc_err = (CI_auc[1]-CI_auc[0])/2

fig, ax = plt.subplots()

rects1 = ax.bar(ind[0] + width/2, acc_mean, width,  yerr=cnn_acc_err,
                label='ACC')
rects2 = ax.bar(ind[1] + width/2, sen_mean, width,  yerr=cnn_sen_err,
                label='SEN')
rects3 = ax.bar(ind[2]+ width/2, spe_mean, width,  yerr=cnn_spe_err,
                label='SPE')
rects4 = ax.bar(ind[3] + width/2, auc_mean, width,  yerr=cnn_auc_err,
                label='AUC')

ax.set_title('ADNI AD-CN')
ax.set_ylim([0.5,1])
ax.grid(axis='y', linestyle='dotted')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xticks(ind+width/2)
ax.set_xticklabels(['ACC','SEN','SPE','AUC'])

autolabel(rects1, "right", 3)
autolabel(rects2, "right", 3)
autolabel(rects3, "right", 3)
autolabel(rects4, "right", 3)

lgd = ax.legend(loc='lower left', bbox_to_anchor=(0, 0), ncol=2)
#save the plot
fig.savefig('.png', dpi=400, facecolor='w', edgecolor='w', bbox_extra_artists=(lgd,), bbox_inches='tight')

        
                   
###draw ROC curve                                      
for i in range(K): 
       plt.plot(fpr_list[i], tpr_list[i], 
         lw=2, alpha=.8,label=r'ROC Curve iteration%d (AUC = %0.2f)' % (i+1,auc_list[i]))
   

plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',label='Chance', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (1-Specifity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right")
plt.title('ROC Curve (AUC = %0.2f $\pm$ %0.3f)' % (auc_mean, auc_std*100))
#save ROC curve
plt.savefig(" .png")
















































