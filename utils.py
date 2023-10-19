import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import os
import sys
import datetime
import csv
import pandas as pd
import h5py
from collections import OrderedDict

import torchvision
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import random

# For visualize input
from torch.utils.tensorboard import SummaryWriter
import io
import torchvision
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):   
  def __init__(self, F_out, outchans, montage, K):
    super().__init__()
    self.D2 = 320
    self.outchans = outchans
    self.spatial_attention = SpatialAttention( outchans, K, montage[:,0], montage[:,1])
    self.conv = nn.Conv2d(outchans, outchans, 1, padding='same')
    self.conv_blocks = nn.Sequential(*[self.generate_conv_block(k) for k in range(5)]) # 5 conv blocks
    self.final_convs = nn.Sequential(
      nn.Conv2d(self.D2, self.D2*2, 1),
      nn.GELU(),
      nn.Conv2d(self.D2*2, F_out, 1)
    )
    self.l1 = nn.Linear(256*F_out, 2)
    
  def generate_conv_block(self, k):
    kernel_size = (1,3)
    padding = 'same' # (p,0)
    return nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(self.outchans if k==0 else self.D2, self.D2, kernel_size, dilation=pow(2,(2*k)%5), padding=padding)),
      ('bn1',   nn.BatchNorm2d(self.D2)), 
      ('gelu1', nn.GELU()),
      ('conv2', nn.Conv2d(self.D2, self.D2, kernel_size, dilation=pow(2,(2*k+1)%5), padding=padding)),
      ('bn2',   nn.BatchNorm2d(self.D2)),
      ('gelu2', nn.GELU()),
      ('conv3', nn.Conv2d(self.D2, self.D2*2, kernel_size, padding=padding)),
      ('glu',   nn.GLU(dim=1))
    ]))

  def forward(self, x):
    
    x = self.spatial_attention(x[:,0]).unsqueeze(2) # add dummy dimension at the end
    x = self.conv(x)
        
    for k in range(len(self.conv_blocks)):
      if k == 0:
        x = self.conv_blocks[k](x)
      else:
        x_copy = x
        for name, module in self.conv_blocks[k].named_modules():
          if name == 'conv2' or name == 'conv3':
            x = x_copy + x # residual skip connection for the first two convs
            x_copy = x.clone() # is it deep copy?
          x = module(x)
    x = self.final_convs(x)
    x = torch.flatten(x, 1)
    x = F.softmax(self.l1(x), -1)
        
    return x    
        
class SpatialAttention(nn.Module):
  def __init__(self, out_channels, K):
    super().__init__()
    self.outchans = out_channels
    self.K = K       
    # trainable parameter:
    self.z = Parameter(torch.randn(self.outchans, K*K, dtype = torch.cfloat,device=device)/(32*32)) # each output channel has its own KxK z matrix
    self.z.requires_grad = True
            
  def forward(self, X):
    cos_mat24, sin_mat24 = compute_cos_sin(positions24[:,0], positions24[:,1], self.K)
    cos_mat128, sin_mat128 = compute_cos_sin(positions128[:,0], positions128[:,1], self.K)
    a_24 = torch.matmul(self.z.real, cos_mat24.T) + torch.matmul(self.z.imag, sin_mat24.T)
    a_128 = torch.matmul(self.z.real, cos_mat128.T) + torch.matmul(self.z.imag, sin_mat128.T)
    sol = []   
    index = random.randint(0, 22)
    x_drop = positions24[index,0]
    y_drop = positions24[index,1]          
    for eeg in X:
        eeg = eeg.to(device=device, dtype=torch.float32)
        if eeg.shape[0] == 23:
            a = a_24
            x_coord = positions24[:,0]
            y_coord = positions24[:,1]
        elif eeg.shape[0] == 128:
            a = a_128
            x_coord = positions128[:,0]
            y_coord = positions128[:,1]
    # Question: divide this with square root of KxK? to stablize gradient as with self-attention?
        for i in range(a.shape[1]):
            distance = (x_drop - x_coord[i])**2 + (y_drop - y_coord[i])**2
            if distance < 0.1:
                a = torch.cat((a[:, :i], a[:, i+1:]), dim = 1)
                eeg = torch.cat((eeg[:i], eeg[i+1:]), dim=0)
        
        a = F.softmax(a, dim=1) # softmax over all input chan location for each output chan
                                            # outchans x  inchans
                
            # X: N x 273 x 360            
        sol.append(torch.matmul(a, eeg)) # N x outchans x 360 (time)
                                   # matmul dim expansion logic: https://pytorch.org/docs/stable/generated/torch.matmul.html
    return torch.stack(sol).to(device=device, dtype=torch.float32)


class EEGDataset(Dataset):
    '''
    Custom Dataset object for PyTorch to load the dataset
    '''
    def __init__(self, x, y, train, val):
        super(EEGDataset).__init__()
        assert x.shape[0] == y.size
        self.x = x
        self.y = [y[i][0] for i in range(y.size)]
        self.train = train
        self.val = val

    def __getitem__(self,key):
        return (self.x[key], self.y[key])
          
    def __len__(self):
        return len(self.y)

class EEGDatasetMixed(Dataset):
    '''
    Custom Dataset object for PyTorch to load the dataset
    '''
    def __init__(self, x:list, y:list, train, val):
        super(EEGDataset).__init__()
        assert len(x) == len(y)
        self.x = x
        self.y = [y[i][0] for i in range(len(y))]
        self.train = train
        self.val = val

    def __getitem__(self,key):
        return (self.x[key], self.y[key])
          
    def __len__(self):
        return len(self.y)


class EEGDataset_Constant(Dataset):
    '''
    Custom Dataset object for PyTorch to load the dataset
    '''
    def __init__(self, x, y, train, val):
        super(EEGDataset).__init__()
        assert x.shape[0] == y.size
        self.x = x
        self.y = [y[i][0] for i in range(y.size)]
        self.train = train
        self.val = val

    def __getitem__(self,key):
        return (self.x[key], self.y[key])
          
    def __len__(self):
        return len(self.y)
           
def load_data(path, role, winLength, numChan, srate, feature, one_channel=False, version=""):
    """
    Load dataset
    :param  
        path: Filepath to the dataset
        role: Role of the dataset. Can be "train", "val", or "test"
        winLength: Length of time window. Can be 2 or 15
        numChan: Number of channels. Can be 24 or 128
        srate: Sampling rate. Supporting 126Hz
        feature: Input feature. Can be "raw", "spectral", or "topo"
        one_channel: Where input has 1 or 3 channel in depth dimension. Matters when load topo data as number of input channels 
                are different from original's
        version: Any additional information of the datafile. Will be appended to the file name at the end
    """
    transform = T.Compose([
        T.ToTensor()
    ])
    if version:
        f = h5py.File(path + "child_mind_abdu/" + f"child_mind_x_{role}_{winLength}s_24chan_{feature}_{version}.mat", 'r')
    else:
        f = h5py.File(path + "child_mind_abdu/" + f"child_mind_x_{role}_{winLength}s_24chan_{feature}.mat", 'r')
    x24 = f[f'X_{role}']    
    if version:
        f = h5py.File(path + "child_mind_abdu128/" + f"child_mind_x_{role}_{winLength}s_128chan_{feature}_{version}.mat", 'r')
    else:
        f = h5py.File(path + "child_mind_abdu128/" + f"child_mind_x_{role}_{winLength}s_128chan_{feature}.mat", 'r')
    x128 = f[f'X_{role}']
    if feature == 'raw':
        x24 = np.transpose(x24,(0,2,1))
        x24 = np.reshape(x24,(-1,1,24,winLength*srate))
        x128 = np.transpose(x128,(0,2,1))
        x128 = np.reshape(x128,(-1,1,128,winLength*srate))
    elif feature == 'topo':
        if one_channel:
            samples = []
            for i in range(x.shape[0]):
                image = x[i]
                b, g, r = image[0,:, :], image[1,:, :], image[2,:, :]
                concat = np.concatenate((b,g,r), axis=1)
                samples.append(concat)
            x = np.stack(samples)
            x = np.reshape(x,(-1,1,x.shape[1],x.shape[2]))
    
    if version:
        f = h5py.File(path + "child_mind_abdu/" + f"child_mind_y_{role}_{winLength}s_24chan_{feature}_{version}.mat", 'r')
    else:
        f = h5py.File(path + "child_mind_abdu/" + f"child_mind_y_{role}_{winLength}s_24chan_{feature}.mat", 'r')
    y24 = f[f'Y_{role}']
    if version:
        f = h5py.File(path + "child_mind_abdu128/" + f"child_mind_y_{role}_{winLength}s_128chan_{feature}_{version}.mat", 'r')
    else:
        f = h5py.File(path + "child_mind_abdu128/" + f"child_mind_y_{role}_{winLength}s_128chan_{feature}.mat", 'r')
    y128 = f[f'Y_{role}']
    return EEGDataset(x24, y24, x128, y128, role=='train', role=='val')


def load_data_constant(path, role, winLength, numChan, srate, feature, one_channel=False, version=""):
    """
    Load dataset
    :param  
        path: Filepath to the dataset
        role: Role of the dataset. Can be "train", "val", or "test"
        winLength: Length of time window. Can be 2 or 15
        numChan: Number of channels. Can be 24 or 128
        srate: Sampling rate. Supporting 126Hz
        feature: Input feature. Can be "raw", "spectral", or "topo"
        one_channel: Where input has 1 or 3 channel in depth dimension. Matters when load topo data as number of input channels 
                are different from original's
        version: Any additional information of the datafile. Will be appended to the file name at the end
    """
    transform = T.Compose([
        T.ToTensor()
    ])
    if version:
        f = h5py.File(path + f"child_mind_x_{role}_{winLength}s_128chan_{feature}_{version}.mat", 'r')
    else:
        f = h5py.File(path + f"child_mind_x_{role}_{winLength}s_128chan_{feature}.mat", 'r')
    x = f[f'X_{role}']
    if feature == 'raw':
        x = np.transpose(x,(0,2,1))
        x = np.reshape(x,(-1,1,128,winLength*srate))
        if numChan == 24:
            x = x[:, :, [22, 9, 33, 24, 11, 124, 122, 29, 6, 111, 45, 36, 104, 108, 42, 55, 93, 58, 52, 62, 92, 96, 70]]
    elif feature == 'topo':
        if one_channel:
            samples = []
            for i in range(x.shape[0]):
                image = x[i]
                b, g, r = image[0,:, :], image[1,:, :], image[2,:, :]
                concat = np.concatenate((b,g,r), axis=1)
                samples.append(concat)
            x = np.stack(samples)
            x = np.reshape(x,(-1,1,x.shape[1],x.shape[2]))
    
    if version:
        f = h5py.File(path + f"child_mind_y_{role}_{winLength}s_128chan_{feature}_{version}.mat", 'r')
    else:
        f = h5py.File(path + f"child_mind_y_{role}_{winLength}s_128chan_{feature}.mat", 'r')
    y = f[f'Y_{role}']
   
    return EEGDataset_Constant(x, y, role=='train', role=='val')


def load_data_mixed(path, role, winLength, numChan, srate, feature, one_channel=False, version=""):
    """
    Load dataset
    :param  
        path: Filepath to the dataset
        role: Role of the dataset. Can be "train", "val", or "test"
        winLength: Length of time window. Can be 2 or 15
        numChan: Number of channels. Can be 24 or 128
        srate: Sampling rate. Supporting 126Hz
        feature: Input feature. Can be "raw", "spectral", or "topo"
        one_channel: Where input has 1 or 3 channel in depth dimension. Matters when load topo data as number of input channels 
                are different from original's
        version: Any additional information of the datafile. Will be appended to the file name at the end
    """
    transform = T.Compose([
        T.ToTensor()
    ])
    if version:
        f = h5py.File(path + f"child_mind_x_{role}_{winLength}s_{numChan}chan_{feature}_{version}.mat", 'r')
    else:
        f = h5py.File(path + f"child_mind_x_{role}_{winLength}s_{numChan}chan_{feature}.mat", 'r')
    x = f[f'X_{role}']
    if feature == 'raw':
        x = np.transpose(x,(0,2,1))
        x = np.reshape(x,(-1,1,numChan,winLength*srate))
    elif feature == 'topo':
        if one_channel:
            samples = []
            for i in range(x.shape[0]):
                image = x[i]
                b, g, r = image[0,:, :], image[1,:, :], image[2,:, :]
                concat = np.concatenate((b,g,r), axis=1)
                samples.append(concat)
            x = np.stack(samples)
            x = np.reshape(x,(-1,1,x.shape[1],x.shape[2]))
    
    if version:
        f = h5py.File(path + f"child_mind_y_{role}_{winLength}s_{numChan}chan_{feature}_{version}.mat", 'r')
    else:
        f = h5py.File(path + f"child_mind_y_{role}_{winLength}s_{numChan}chan_{feature}.mat", 'r')
    y = f[f'Y_{role}']
    
    chan_128_X, chan_128_Y, chan_23_X, chan_23_Y = data_subsampling(x, y)
    
    X_mixed = chan_128_X
    X_mixed.extend(chan_23_X)
    Y_mixed = chan_128_Y
    Y_mixed.extend(chan_23_Y)
    
    return EEGDatasetMixed(X_mixed, Y_mixed, role=='train', role=='val')


def get_subjects_start_indices(labels):
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Not binary classes 0 - 1')
    cur_label = labels[0]
    subject_start_indices = {"0": [], "1": []}
    subject_start_indices[str(int(cur_label))].append(0)
    for i in range(1,len(labels)):
        if labels[i] == cur_label:
            continue
        else:
            subject_start_indices[str(int(labels[i]))].append(i)
            cur_label = labels[i]
    return subject_start_indices

def data_subsampling(X, Y):
    subject_start_indices = get_subjects_start_indices(Y)
    min_classsize = np.min([len(subject_start_indices['0']), len(subject_start_indices['1'])])
    classA_start_indices = np.array(subject_start_indices['0'][:min_classsize])
    classB_start_indices = np.array(subject_start_indices['1'][:min_classsize])
    all_start_indices = subject_start_indices['0']
    all_start_indices.extend(subject_start_indices['1'])
    min_subj_nsample = np.min(np.diff(sorted(all_start_indices)))
    
    half_size = int(min_classsize/2)
    chan_128_X = []
    chan_23_X = []
    chan_128_Y = []
    chan_23_Y = []
    sub_chan_indices = [22, 9, 33, 24, 11, 124, 122, 29, 6, 111, 45, 36, 104, 108, 42, 55, 93, 58, 52, 62, 92, 96, 70]
    for i in range(half_size):
        chan_128_Y.extend(Y[classA_start_indices[i]:classA_start_indices[i]+min_subj_nsample])
        chan_128_Y.extend(Y[classB_start_indices[i]:classB_start_indices[i]+min_subj_nsample])

        chan_23_Y.extend(Y[classA_start_indices[i+half_size]:classA_start_indices[i+half_size]+min_subj_nsample])
        chan_23_Y.extend(Y[classB_start_indices[i+half_size]:classB_start_indices[i+half_size]+min_subj_nsample])

        chan_128_X.extend(X[classA_start_indices[i]:classA_start_indices[i]+min_subj_nsample])
        chan_128_X.extend(X[classB_start_indices[i]:classB_start_indices[i]+min_subj_nsample])

        chan_23_X.extend(X[classA_start_indices[i+half_size]:classA_start_indices[i+half_size]+min_subj_nsample][:,:,sub_chan_indices,:])
        chan_23_X.extend(X[classB_start_indices[i+half_size]:classB_start_indices[i+half_size]+min_subj_nsample][:,:,sub_chan_indices,:])
    
    if np.diff(np.unique(chan_128_Y, return_counts=True)[1]) != 0 or np.diff(np.unique(chan_23_Y, return_counts=True)[1]) != 0:
        raise ValueError('Class still unbalanced')
        
    return chan_128_X, chan_128_Y, chan_23_X, chan_23_Y
    
def create_original_model(feature):
    if feature == 'raw':
        model = nn.Sequential(
            nn.Conv2d(1,100,3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(100,100,3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(100,300,(2,3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(300,300,(1,7)),
            nn.ReLU(),
            nn.MaxPool2d((1,2), stride=1),
            nn.Dropout(0.25),
            nn.Conv2d(300,100,(1,3)),
            nn.ReLU(),
            nn.Conv2d(100,100,(1,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1900,6144),
            nn.ReLU(),
            nn.Linear(6144,2),
        )
    elif feature == 'topo':
        model = nn.Sequential()
        model.add_module('convolution', nn.Sequential(
            nn.Conv2d(1,100,3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(100,100,3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(100,300,(2,3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(300,300,(1,7),padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1,2), stride=1),
            nn.Dropout(0.25),
            nn.Conv2d(300,100,(1,3),padding=1),
            nn.ReLU(),
            nn.Conv2d(100,100,(1,3),padding=1),
            nn.ReLU(),
        ))
        model.add_module('dense', nn.Sequential(
            nn.Flatten(),
            nn.Linear(1400,6144),
            nn.ReLU(),
            nn.Linear(6144,2)
        ))
    return model

class VGG_Attention(nn.Module):
  def __init__(self, outchans, K, vgg):
    super().__init__()
    self.spatial_attention = SpatialAttention(outchans, K)
    self.model = vgg
  def forward(self, x):
    x = self.spatial_attention(x).unsqueeze(1)
    return self.model(x)

def compute_cos_sin(x, y, K):
    kk = torch.arange(1, K+1, device=device)
    ll = torch.arange(1, K+1, device=device)
    cos_fun = lambda k, l, x, y: torch.cos(2*torch.pi*(k*x + l*y))
    sin_fun = lambda k, l, x, y: torch.sin(2*torch.pi*(k*x + l*y))
    return torch.stack([cos_fun(kk[None,:], ll[:,None], x, y) for x, y in zip(x, y)]).reshape(x.shape[0],-1).float(), torch.stack([sin_fun(kk[None,:], ll[:,None], x, y) for x, y in zip(x, y)]).reshape(x.shape[0],-1).float()

def create_vgg_rescaled(subsample, channels, time):
    tmp = models.vgg16()
    tmp.features = tmp.features[0:17]
    vgg16_rescaled = nn.Sequential()
    modules = []
    
    first_in_channels = 1
    first_in_features = channels*time
        
    for layer in tmp.features.children():
        if isinstance(layer, nn.Conv2d):
            if layer.in_channels == 3:
                in_channels = first_in_channels
            else:
                in_channels = int(layer.in_channels/subsample)
            out_channels = int(layer.out_channels/subsample)
            modules.append(nn.Conv2d(in_channels, out_channels, layer.kernel_size, layer.stride, layer.padding))
        else:
            modules.append(layer)
    vgg16_rescaled.add_module('features',nn.Sequential(*modules))
    vgg16_rescaled.add_module('flatten', nn.Flatten())

    modules = []
    for layer in tmp.classifier.children():
        if isinstance(layer, nn.Linear):
            if layer.in_features == 25088:
                in_features = first_in_features
            else:
                in_features = int(layer.in_features/subsample) 
            if layer.out_features == 1000:
                out_features = 2
            else:
                out_features = int(layer.out_features/subsample) 
            modules.append(nn.Linear(in_features, out_features))
        else:
            modules.append(layer)
    vgg16_rescaled.add_module('classifier', nn.Sequential(*modules))
    return vgg16_rescaled

positions128 = torch.tensor(np.array([[ 7.15185774e-01,  2.48889351e-01],
       [ 7.02434125e-01,  3.17666701e-01],
       [ 6.95043285e-01,  3.81730395e-01],
       [ 6.64632066e-01,  4.15512025e-01],
       [ 6.17343244e-01,  4.52640363e-01],
       [ 5.70554706e-01,  4.86034445e-01],
       [ 5.21555270e-01,  5.11613217e-01],
       [ 7.77079001e-01,  3.33967050e-01],
       [ 7.58449443e-01,  4.04282683e-01],
       [ 7.24088505e-01,  4.36604693e-01],
       [ 6.85433073e-01,  4.86034445e-01],
       [ 6.17343244e-01,  5.19428527e-01],
       [ 5.59172750e-01,  5.38580522e-01],
       [ 8.18418971e-01,  4.41881185e-01],
       [ 7.60133440e-01,  4.86034445e-01],
       [ 7.35627966e-01,  4.86034445e-01],
       [ 8.45364169e-01,  4.86034445e-01],
       [ 7.24088505e-01,  5.35464197e-01],
       [ 6.64632066e-01,  5.56556865e-01],
       [ 6.08030772e-01,  5.74970128e-01],
       [ 8.18418971e-01,  5.30187705e-01],
       [ 7.58449443e-01,  5.67786207e-01],
       [ 6.95043285e-01,  5.90338495e-01],
       [ 6.39581248e-01,  5.97577739e-01],
       [ 7.77079001e-01,  6.38101840e-01],
       [ 7.02434125e-01,  6.54402189e-01],
       [ 6.36080215e-01,  6.39322188e-01],
       [ 5.90689607e-01,  6.21729782e-01],
       [ 5.49930209e-01,  5.93637909e-01],
       [ 5.09601084e-01,  5.65410534e-01],
       [ 4.74660346e-01,  5.26921639e-01],
       [ 7.15185774e-01,  7.23179539e-01],
       [ 6.27432502e-01,  7.00712475e-01],
       [ 5.70168887e-01,  6.72695237e-01],
       [ 5.24983262e-01,  6.41061825e-01],
       [ 4.95401265e-01,  6.09669558e-01],
       [ 4.56179857e-01,  5.70442673e-01],
       [ 6.52314440e-01,  7.82311284e-01],
       [ 5.18992954e-01,  7.39998487e-01],
       [ 4.89867494e-01,  6.91482721e-01],
       [ 4.69348952e-01,  6.55201509e-01],
       [ 4.34670981e-01,  6.24309562e-01],
       [ 6.28180126e-01,  8.61450930e-01],
       [ 5.49739502e-01,  8.11172169e-01],
       [ 4.24802980e-01,  7.37204171e-01],
       [ 4.25680921e-01,  6.87383067e-01],
       [ 4.13285406e-01,  6.55344129e-01],
       [ 6.71360349e-01,  9.72068890e-01],
       [ 5.16948449e-01,  9.08970332e-01],
       [ 3.55532101e-01,  7.10977843e-01],
       [ 3.62120543e-01,  6.64285008e-01],
       [ 3.79156415e-01,  6.28501244e-01],
       [ 3.98210730e-01,  5.80314761e-01],
       [ 4.15632463e-01,  5.34804692e-01],
       [ 4.44251810e-01,  4.86034445e-01],
       [ 3.37112089e-01,  8.67523508e-01],
       [ 3.26773432e-01,  7.71448657e-01],
       [ 3.02409518e-01,  6.81637463e-01],
       [ 3.15334354e-01,  6.26408474e-01],
       [ 3.42865950e-01,  5.85266696e-01],
       [ 3.65550361e-01,  5.38836376e-01],
       [ 3.34656872e-01,  4.86034445e-01],
       [ 2.28916019e-01,  8.08418208e-01],
       [ 2.40816811e-01,  7.14482456e-01],
       [ 2.54299126e-01,  6.35765016e-01],
       [ 2.82959322e-01,  5.83008874e-01],
       [ 3.09968299e-01,  5.30931543e-01],
       [ 1.33841406e-01,  6.96028404e-01],
       [ 1.74061857e-01,  6.32194751e-01],
       [ 2.13486999e-01,  5.73685443e-01],
       [ 2.63433137e-01,  5.23590932e-01],
       [ 2.93992338e-01,  4.86034445e-01],
       [ 8.69873448e-02,  5.89501183e-01],
       [ 1.46141772e-01,  5.31677906e-01],
       [ 2.05894177e-01,  4.86034445e-01],
       [ 2.63433137e-01,  4.48477958e-01],
       [ 3.09968299e-01,  4.41137347e-01],
       [ 3.65550361e-01,  4.33232515e-01],
       [ 4.15632463e-01,  4.37264198e-01],
       [ 4.74660346e-01,  4.45147251e-01],
       [ 7.87721494e-02,  4.86034445e-01],
       [ 1.46141772e-01,  4.40390984e-01],
       [ 2.13486999e-01,  3.98383447e-01],
       [ 2.82959322e-01,  3.89060016e-01],
       [ 3.42865950e-01,  3.86802194e-01],
       [ 3.98210730e-01,  3.91754129e-01],
       [ 4.56179857e-01,  4.01626217e-01],
       [ 8.69873448e-02,  3.82567707e-01],
       [ 1.74061857e-01,  3.39874139e-01],
       [ 2.54299126e-01,  3.36303874e-01],
       [ 3.15334354e-01,  3.45660416e-01],
       [ 3.79156415e-01,  3.43567646e-01],
       [ 4.34670981e-01,  3.47759328e-01],
       [ 1.33841406e-01,  2.76040486e-01],
       [ 2.40816811e-01,  2.57586434e-01],
       [ 3.02409518e-01,  2.90431427e-01],
       [ 3.62120543e-01,  3.07783882e-01],
       [ 4.13285406e-01,  3.16724761e-01],
       [ 2.28916019e-01,  1.63650682e-01],
       [ 3.26773432e-01,  2.00620233e-01],
       [ 3.55532101e-01,  2.61091048e-01],
       [ 4.25680921e-01,  2.84685823e-01],
       [ 4.69348952e-01,  3.16867381e-01],
       [ 4.95401265e-01,  3.62399332e-01],
       [ 5.09601084e-01,  4.06658357e-01],
       [ 5.21555270e-01,  4.60455673e-01],
       [ 3.37112089e-01,  1.04545383e-01],
       [ 4.24802980e-01,  2.34864719e-01],
       [ 4.89867494e-01,  2.80586169e-01],
       [ 5.24983262e-01,  3.31007065e-01],
       [ 5.49930209e-01,  3.78430981e-01],
       [ 5.59172750e-01,  4.33488368e-01],
       [ 5.16948449e-01,  6.30985576e-02],
       [ 5.49739502e-01,  1.60896721e-01],
       [ 5.18992954e-01,  2.32070403e-01],
       [ 5.70168887e-01,  2.99373653e-01],
       [ 5.90689607e-01,  3.50339108e-01],
       [ 6.08030772e-01,  3.97098762e-01],
       [ 6.71360349e-01, -5.20417043e-18],
       [ 6.28180126e-01,  1.10617960e-01],
       [ 6.52314440e-01,  1.89757606e-01],
       [ 6.27432502e-01,  2.71356415e-01],
       [ 6.36080215e-01,  3.32746702e-01],
       [ 6.39581248e-01,  3.74491151e-01],
       [ 7.15997599e-01,  1.78986800e-01],
       [ 9.02096460e-01,  2.53445652e-01],
       [ 9.02096460e-01,  7.18623238e-01],
       [ 7.15997599e-01,  7.93082090e-01]]))

positions24 = positions128[[22, 9, 33, 24, 11, 124, 122, 29, 6, 111, 45, 36, 104, 108, 42, 55, 93, 58, 52, 62, 92, 96, 70]]

def create_model(model_type):
    if model_type == 'vgg':
        return create_vgg_rescaled(4, 128, 256)
    elif model_type == 'vgg_attention':
        vgg_model = create_vgg_rescaled(4, 128, 256)
        return VGG_Attention(128, 32, vgg_model)
    else:
        return Net(F_out = 120, outchans = 270, K = 32)

def check_accuracy(loader, model, device, dtype):
    '''
    Check accuracy of the model 
    param:
        loader: An EEGDataset object
        model: A PyTorch Module to test
        device: cpu or cuda
        dtype: value type
        logger: Logger object for logging purpose
    '''
    
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            #x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        return acc

def train(model, loader_train, loader_val, optimizer, epochs):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    - logger: Logger object for logging purpose
    Returns: Nothing, but prints model accuracies during training.
    """
    dtype = torch.float32
    model.train()  # put model to training mode
    loss_t = []
    for e in range(epochs):
        print(f'epoch {e}')
        loss_train = 0
        for t, (x, y) in enumerate(loader_train):

            #x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            loss_train = loss_train + loss

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            
        total_sum = 0
        with torch.no_grad():
            for params in model.parameters():
                total_sum += torch.sum(params)
        print('model weights', total_sum)
        print('train loss', loss_train)
        loss_t.append(loss_train)

        train_acc = check_accuracy(loader_train, model, device, dtype)
        print('Train Accuracy at Epoch ' + str(e) + ': ' + str(train_acc)) 
        val_acc = check_accuracy(loader_val, model, device, dtype)
        print('Val Accuracy at Epoch ' + str(e) + ': ' + str(val_acc))

    print(loss_t)

    return model

def run_experiment(seed, loader_train, loader_val, loader_test, num_epoch):
    model = create_model('vgg_attention')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    dtype = torch.float32

    np.random.seed(seed)
    torch.manual_seed(seed)

    # toggle between learning rate and batch size values 
    
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, weight_decay=0.001)
    model = train(model, loader_train, loader_val, optimizer, epochs=num_epoch)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_acc = check_accuracy(loader_test, model, device, dtype)
    print("Test Accuracy: " + str(test_acc))

    path = '/expanse/projects/nsg/external_users/public/arno/child_mind_abdu128/'
    winLength = 2
    numChan = 24
    srate = 128
    feature = 'raw'
    one_channel = False
    
    # Testing
    return model

def test_model(model, test_data, subj_csv, device, dtype):
    # one-segment test
    loader_test = DataLoader(test_data, batch_size=70)
    per_sample_acc = check_accuracy(loader_test, model, device, dtype)

    # 40-segment test
    with open(subj_csv, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        subjIDs = [row[0] for row in spamreader]
    unique_subjs,indices = np.unique(subjIDs,return_index=True)

    iterable_test_data = list(iter(DataLoader(test_data, batch_size=1)))
    num_correct = []
    for subj,idx in zip(unique_subjs,indices):
    #     print(f'Subj {subj} - gender {iterable_test_data[idx][1]}')
        data = iterable_test_data[idx:idx+40]
        #print(np.sum([y for _,y in data]))
        assert 40 == np.sum([y for _,y in data]) or 0 == np.sum([y for _,y in data])
        preds = []
        correct = 0
        with torch.no_grad():
            for x,y in data:
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                correct = y
                scores = model(x)
                _, pred = scores.max(1)
                preds.append(pred)
        final_pred = (torch.mean(torch.FloatTensor(preds)) > 0.5).sum()
        num_correct.append((final_pred == correct).sum())
    #print(len(num_correct))
    acc = float(np.sum(num_correct)) / len(unique_subjs)
    return per_sample_acc, acc


def test_all_seeds(model_path, model_type, feature, test_data, subjIDs_file, epoch, num_seed, device, dtype, logger):
    sample_acc = []
    subject_acc = []
    for s in range(num_seed):
        model = create_model(model_type, feature)
        model.load_state_dict(torch.load(f'{model_path}-seed{s}-epoch{epoch}'))
        model.to(device=device)
        sam_acc, sub_acc = test_model(model, test_data,subjIDs_file, device, dtype, logger)
        sample_acc.append(sam_acc)
        subject_acc.append(sub_acc)
        
    sample_acc = np.multiply(sample_acc,100)
    subject_acc = np.multiply(subject_acc,100)
    return sample_acc, subject_acc

def get_stats(arr):
    return np.min(arr), np.max(arr), np.mean(arr), np.std(arr)
