
from sklearn import metrics
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import nibabel as nib
import itertools

idx_to_label = {0: 'non-DR', 1: 'NPDR', 2: 'PDR'}


def getKey(item):
    for idx, label in idx_to_label.items():
        if label == item:
            return idx


def getValue(key):
    for idx, label in idx_to_label.items():
        if idx == key:
            return label



def sample_loss(model,mood_test,y_valid,device, thresh):
    pred = []
    model.eval()
    for X in mood_test:
        x_img = torch.from_numpy(X.copy()).to(device).float()
        with torch.no_grad():
            codes = model.retrieve_codes(x_img).flatten(1)
            loss = model.loss(x_img, reduction='none')['loss'].flatten(1)
            score1 = torch.sum(loss*(loss>thresh),1).float()
            pred.append([score1.cpu().numpy()])
    pred = np.concatenate(pred,1).reshape(-1,1)
    
    avg =[]
    classes = idx_to_label.values()
    normal_loss = pred[np.where(y_valid == 0)[0]]
    normal_loss_avg = np.sum(normal_loss)/normal_loss.shape[0]
    avg.append(np.ravel(normal_loss).tolist())
    cnv_loss = pred[np.where(y_valid == 1)[0]]
    cnv_loss_avg = np.sum(cnv_loss)/cnv_loss.shape[0]
    avg.append(np.ravel(cnv_loss).tolist())
    dr_loss = pred[np.where(y_valid == 2)[0]]
    dr_loss_avg = np.sum(dr_loss)/dr_loss.shape[0]
    avg.append(np.ravel(dr_loss).tolist())
    plt.figure(figsize=(9, 7))
    plt.boxplot(avg,labels=classes)
    plt.xlabel("Classes")
    plt.ylabel("Loss value")
    plt.title("Loss per class-validation")
    plt.show()
    
    
    y_test_2 = np.where(y_valid > 0, 1, y_valid)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test_2, pred)
    roc_auc_all = metrics.auc(fpr, tpr)
    
    
    
    
    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,color="darkorange",lw=lw,label="ROC curve (area = %0.2f)" % roc_auc_all,)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (Healty, Unhealthy) validation")
    plt.legend(loc="lower right")
    plt.show()
    
    
    
    normal_loss = pred[np.where(y_valid == 0)[0]]
    normal_label = y_valid[np.where(y_valid == 0)[0]]

    amd_loss = pred[np.where(y_valid == 1)[0]]
    amd_label = y_valid[np.where(y_valid == 1)[0]]

    loss_nor = np.concatenate((normal_loss, amd_loss), axis=0)

    X_test_nor = np.concatenate((normal_label, amd_label), axis=0)
    X_test_nor = np.where(X_test_nor > 0, 1, X_test_nor)
    fpr, tpr, thresholds = metrics.roc_curve(X_test_nor, loss_nor/400)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,color="darkorange",lw=lw,label="ROC curve (area = %0.2f)" % roc_auc,)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (Healty, NPDR) - validation")
    plt.legend(loc="lower right")
    plt.show()
    
    
    normal_loss = pred[np.where(y_valid == 0)[0]]
    normal_label = y_valid[np.where(y_valid == 0)[0]]

    amd_loss = pred[np.where(y_valid == 2)[0]]
    amd_label = y_valid[np.where(y_valid == 2)[0]]

    loss_nor = np.concatenate((normal_loss, amd_loss), axis=0)

    X_test_nor = np.concatenate((normal_label, amd_label), axis=0)
    X_test_nor = np.where(X_test_nor > 0, 1, X_test_nor)
    fpr, tpr, thresholds = metrics.roc_curve(X_test_nor, loss_nor/400)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,color="darkorange",lw=lw,label="ROC curve (area = %0.2f)" % roc_auc,)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (Healty, PDR) - validation")
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc_all