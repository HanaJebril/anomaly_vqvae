from pathlib import Path
import nibabel as nib
import numpy as np
from collections import namedtuple
import torch
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader
from skimage.filters import threshold_mean
from skimage.filters import sato, median
from skimage.morphology import area_opening
import os
import shutil
import natsort
from seeds_file import seedEverything
seedEverything(43)


def image_segmentation(image):
    ground_truth = np.array(image)
    remove_salt = median(ground_truth)
    morph_filter = area_opening(remove_salt)
    morph_filter = median(morph_filter)
    kwargs = {'sigmas': [1], 'mode': 'reflect', 'black_ridges':False}
    filter_res = sato(morph_filter, **kwargs)
    thresh = threshold_mean(filter_res)
    binary = filter_res > thresh
    return binary



def cal_Dice(img1,img2):
    shape = img1.shape
    I = 0
    U = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] >= 1 and img2[i,j] >= 1:
                I += 1
            if img1[i,j] >= 1 or img2[i,j] >= 1:
                U += 1
    if (U==0):
        return 1
    return 2*I/(I+U+1e-5)


idx_to_label = {0: 'NORMAL', 1: 'CNV', 2: 'DR', 3: 'AMD',4: 'RVO', 5: 'OTHERS',6:'CSC'}


def getKey(item):
    for idx, label in idx_to_label.items():
        if label == item:
            return idx


def getValue(key):
    for idx, label in idx_to_label.items():
        if idx == key:
            return label

def validation_batch(img, label):
    datasize = img.shape[1]
    blocksize = 152
    batch_size = 4
    val_images =  np.zeros((img.shape[0],batch_size, blocksize, blocksize))
    annotations = np.zeros((img.shape[0],batch_size))
    idx = 0
    for i in range(0, datasize, blocksize):
        for j in range(0, datasize, blocksize):
            val_images[:, idx, 0:blocksize, 0:blocksize] = img[:, i:i + blocksize,j:j + blocksize]
            annotations[:, idx] = label
            idx+=1
    image = torch.from_numpy(val_images)
    labels = torch.from_numpy(annotations)
    images = torch.reshape(image, (image.shape[0]*image.shape[1], image.shape[2],image.shape[3]))
    label = torch.reshape(labels,(-1,))
    # plt.figure(figsize=(15, 9))
    fig, axs = plt.subplots(1, 4, figsize=(12, 6))                    
    ax = axs[0]
    axs[0].imshow(images[0,:,:].cpu().numpy(), cmap="gray")
    axs[1].imshow(images[1,:,:].cpu().numpy(), cmap="gray")
    axs[2].imshow(images[2,:,:].cpu().numpy() ,cmap="gray")
    axs[3].imshow(images[3,:,:].cpu().numpy() ,cmap="gray")
    plt.tight_layout()
    plt.show()  
    return images, label



def read_batch_normal_train(img, label):#FAZ segmentation
    batch_size = 10
    datasize = img.shape[1]
    blocksize = 152
    images = np.zeros((img.shape[0],batch_size, blocksize, blocksize))
    annotations = np.zeros((img.shape[0],batch_size))

    sd=50 #Standard Deviation
    for batch in range(0,batch_size):
        nx=int(np.random.normal(datasize/2,sd))
        ny=int(np.random.normal(datasize/2,sd))
        startx = nx-int(blocksize/2)
        endx = nx+int(blocksize/2)
        starty= ny-int(blocksize/2)
        endy = ny + int(blocksize/2)
        while startx<0 or starty<0 or endx>datasize or endy>datasize:
        # nx = np.random.randint(blocksize/2,datasize-blocksize/2)
        # ny = np.random.randint(blocksize/2,datasize-blocksize/2)
            nx=int(np.random.normal(datasize/2,sd))
            ny=int(np.random.normal(datasize/2,sd))
            startx = nx - int(blocksize/ 2)
            endx = nx + int(blocksize/ 2)
            starty = ny - int(blocksize/ 2)
            endy = ny + int(blocksize/ 2)
        images[:,batch, 0:blocksize, 0:blocksize] = img[:, startx:endx, starty:endy]
        annotations[:, batch] = label
    image = torch.from_numpy(images)
    labels = torch.from_numpy(annotations)
    images = torch.reshape(image, (image.shape[0]*image.shape[1], image.shape[2],image.shape[3]))
    label = torch.reshape(labels,(-1,))
    # plt.figure(figsize=(15, 9))
    fig, axs = plt.subplots(1, 5, figsize=(12, 6))                    
    ax = axs[0]
    axs[0].imshow(images[0,:,:].cpu().numpy(), cmap="gray")
    axs[1].imshow(images[1,:,:].cpu().numpy(), cmap="gray")
    axs[2].imshow(images[2,:,:].cpu().numpy() ,cmap="gray")
    axs[3].imshow(images[3,:,:].cpu().numpy() ,cmap="gray")
    axs[4].imshow(images[4,:,:].cpu().numpy() ,cmap="gray")
    plt.tight_layout()
    plt.show()  
    return images, label


def read_batch_random_train(img, label):#FAZ segmentation
    batch_size = 10
    datasize = img.shape[1]
    blocksize = 152
    images = np.zeros((img.shape[0],batch_size, blocksize, blocksize))
    annotations = np.zeros((img.shape[0],batch_size))

    sd=50 #Standard Deviation
    for batch in range(0,batch_size):
        nx = np.random.randint(blocksize/2,datasize-blocksize/2)
        ny = np.random.randint(blocksize/2,datasize-blocksize/2)
        startx = nx-int(blocksize/2)
        endx = nx+int(blocksize/2)
        starty= ny-int(blocksize/2)
        endy = ny + int(blocksize/2)
        while startx<0 or starty<0 or endx>datasize or endy>datasize:
            nx = np.random.randint(blocksize/2,datasize-blocksize/2)
            ny = np.random.randint(blocksize/2,datasize-blocksize/2)
            startx = nx - int(blocksize/ 2)
            endx = nx + int(blocksize/ 2)
            starty = ny - int(blocksize/ 2)
            endy = ny + int(blocksize/ 2)
        images[:,batch, 0:blocksize, 0:blocksize] = img[:, startx:endx, starty:endy]
        annotations[:, batch] = label
    image = torch.from_numpy(images)
    labels = torch.from_numpy(annotations)
    images = torch.reshape(image, (image.shape[0]*image.shape[1], image.shape[2],image.shape[3]))
    label = torch.reshape(labels,(-1,))
    # plt.figure(figsize=(15, 9))
    fig, axs = plt.subplots(1, 5, figsize=(12, 6))                    
    ax = axs[0]
    axs[0].imshow(images[0,:,:].cpu().numpy(), cmap="gray")
    axs[1].imshow(images[1,:,:].cpu().numpy(), cmap="gray")
    axs[2].imshow(images[2,:,:].cpu().numpy() ,cmap="gray")
    axs[3].imshow(images[3,:,:].cpu().numpy() ,cmap="gray")
    axs[4].imshow(images[4,:,:].cpu().numpy() ,cmap="gray")
    plt.tight_layout()
    plt.show()  
    return images, label





def train(model, train_loader, optimizer, device):
    model.train()
    train_losses = []
    batch_sizes = []
    for img, label in train_loader:
#         img = torch.from_numpy(img)
        loss = model.loss(img.to(device))
        optimizer.zero_grad()

        loss['loss'].backward()

        # Gradient clippling
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.)

        optimizer.step()
        train_losses.append(loss['loss'].item() * img.shape[0])
        batch_sizes.append(img.shape[0])

    return sum(train_losses)/sum(batch_sizes)




def eval_loss(model, data_loader, device,epoch,model_name):
    model.eval()
    eval_losses = []
    batch_sizes = []
    eval_pers = []
    batch_len=[]
    eval_dice =  []
    dice_batch = []
    with torch.no_grad():
        for img, label in data_loader:
#             img = torch.from_numpy(img)
            
            loss = model.loss(img.to(device))
            
            
            eval_losses.append(loss['loss'].item() * img.shape[0])
            batch_sizes.append(img.shape[0])
            batch_len.append(1)
#             for idx, single_img in enumerate(img):
#                 eval_dice.append(cal_Dice(image_segmentation(single_img),image_segmentation(loss['x_tilde'].cpu().data[idx,0] )))
#             if ((epoch+1) % 50 ==0):
#                 if model_name == "vqvae":
#                     orig = img[0]
#                     fig, ax = plt.subplots(1,2)
#                     ax[0].imshow(image_segmentation(orig), cmap=plt.cm.gray)
#                     ax[0].axis('off')
#                     ax[0].set_title(getValue(label[0])+' Original-Segmented')
#                     x=(image_segmentation(loss['x_tilde'].cpu().data[0,0]))
#                     ax[1].imshow(x, cmap=plt.cm.gray)
#                     ax[1].axis('off')
#                     ax[1].set_title(getValue(label[0])+' Rconstructed-Segmented')
#                     plt.show()
                    
                        
# #                     return sum(eval_losses)/sum(batch_sizes), sum(eval_dice)/sum(batch_sizes)
                    
#                 elif model_name== "AR":
#                     orig = loss['code'].cpu().data[0]
# #                 fig, ax = plt.subplots(1,2)
#                 ax[0].imshow(orig, cmap=plt.cm.gray)
#                 ax[0].axis('off')
#                 ax[0].set_title('Original')
#                 x=(loss['x_tilde'].cpu().data[0,0])
#                 ax[1].imshow(x, cmap=plt.cm.gray)
#                 ax[1].axis('off')
#                 ax[1].set_title('Rconstructed')
#                 plt.show()
    if model_name == "vqvae":
        return sum(eval_losses)/sum(batch_sizes), sum(eval_dice)/sum(batch_sizes)
    elif model_name== "AR":
        print(np.percentile(loss['code'].cpu().numpy(), 98))
        eval_pers.append(np.percentile(loss['code'].cpu().numpy(), 98))
        return sum(eval_losses)/sum(batch_sizes), sum(eval_pers)/sum(batch_sizes)

def save_checkpoint(model,optimizer, tracker, file_name):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'tracker': tracker,
    }

    torch.save(checkpoint, file_name)

class train_tracker:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.test_dices = []
        self.lr = []

    def __len__(self):
        return len(self.train_losses)

    def append(self,train_loss,test_loss,lr,test_dice=None):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.test_dices.append(test_dice)
        self.lr.append(lr)

    def plot(self,N=None):
        N = N if N is not None else self.__len__()
        plt.plot(self.train_losses,label='Train')
        plt.plot(self.test_losses, label='Eval')
        plt.legend()
        plt.savefig('Loss_'+str(time.time())+'.png')
        # plt.show()
    def plot_dice(self,N=None):
        N = N if N is not None else self.__len__()
        plt.plot(self.test_dices[-N:])
        plt.title('Dice Score - Validation')
        plt.savefig('Loss_'+str(time.time())+'.png')

def train_epochs(model, optimizer,tracker, train_loader, test_loader, model_name, epochs, device, chpt = None):
    # Early stopping
    the_last_loss = 100
    the_last_dice=0
    patience = 3
    trigger_times = 0
    delta = 0.01
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer,device)
        if model_name == "vqvae":
            test_loss,test_dice = eval_loss(model, test_loader, device,epoch,model_name)
            tracker.append(train_loss,test_loss,optimizer.param_groups[0]['lr'],test_dice = test_dice )
            print('{} epochs, {:.3f} test dice ,{:.3f} test loss, {:.3f} train loss'.format(len(tracker),test_dice, test_loss, train_loss))
            
#             if test_dice > the_last_dice :
#                 save_checkpoint(model,optimizer,tracker,
#                             'dice/{}.pt'.format(test_dice))
#                 the_last_dice = test_dice
                
#             if the_last_loss >= test_loss :
#                 save_checkpoint(model,optimizer,tracker,
#                             'loss/{:.5f}.pt'.format(test_dice))

#                 model_names = natsort.natsorted(
#                     os.listdir('loss'))
#                 #print(len(model_names))
#                 if len(model_names) == 4:
#                     os.remove(
#                         os.path.join('loss', model_names[0]))
#                 the_last_loss = test_loss


            
        else:
            test_loss,test_pers = eval_loss(model, test_loader, device,epoch,model_name)
            tracker.append(train_loss,test_loss,optimizer.param_groups[0]['lr'])
            print('{} epochs, {:.3f} test loss,  {:.3f} test Percentile, {:.3f} train loss'.format(len(tracker), test_loss, test_pers,train_loss))
            
            
            
        if the_last_loss >= test_loss :
            print('loss decreased')
            save_checkpoint(model,optimizer,tracker,
                            'earlyStop/drac/{}/{:.5f}.pt'.format(model_name,test_loss))
            
            model_names = natsort.natsorted(
                    os.listdir('earlyStop/drac/{}'.format(model_name)))
                #print(len(model_names))
            if len(model_names) == 4:
                os.remove(
                    os.path.join('earlyStop/drac/{}'.format(model_name), model_names[-1]))
            the_last_loss = test_loss
           
        
    return model


def load_cid(cid,path):
    """Load segmentation and volume"""
    vol = nib.load(path+'/case_{:05d}/imaging.nii.gz'.format(cid))
    seg = nib.load(path+'/case_{:05d}/segmentation.nii.gz'.format(cid))
    spacing = vol.affine
    vol = np.asarray(vol.get_fdata())
    seg = np.asarray(seg.get_fdata())
    seg = seg.astype(np.int8)
    vol = normalize(vol)
    return vol, seg, spacing


img_extended = namedtuple('img_extended',('img','seg','k','t','coord','cid'))

def get_full_case_id(cid):
    try:
        cid = int(cid)
        case_id = "case_{:05d}".format(cid)
    except ValueError:
        case_id = cid

    return case_id


def get_case_path(cid):
    # Resolve location where data should be living
    data_path = Path(__file__).parent.parent / "data"
    if not data_path.exists():
        raise IOError(
            "Data path, {}, could not be resolved".format(str(data_path))
        )

    # Get case_id from provided cid
    case_id = get_full_case_id(cid)

    # Make sure that case_id exists under the data_path
    case_path = data_path / case_id
    if not case_path.exists():
        raise ValueError(
            "Case could not be found \"{}\"".format(case_path.name)
        )

    return case_path

def dice_score(trues, preds):
    """Calculate dice score / f1 given binary boolean variables: 2 x IoU"""
    return 2. * (trues & preds).sum()/(trues.sum() + preds.sum())

def max_score(trues, pred, score_func = dice_score, steps = 8):
    """Iterate through possible threshold ranges and return max score and argmax threshold """
    min_d, max_d = pred.min(), pred.max()

    for i in range(steps):
        mid_d = (max_d-min_d)/2 + min_d
        mid_s = score_func(trues,pred > mid_d)

        q1_s = score_func(trues,pred > (max_d-min_d)/4 + min_d)
        q3_s = score_func(trues,pred > 3*(max_d-min_d)/4 + min_d)

        if q1_s == q3_s:
            break
        elif q1_s > q3_s:
            max_d = mid_d
        else:
            min_d = mid_d
    return mid_s, mid_d



