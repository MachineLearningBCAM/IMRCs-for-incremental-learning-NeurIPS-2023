#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 08:59:26 2022

@author: 
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from scipy.io import savemat


mat_contents = sio.loadmat('dataset.mat')

# Training and test sets divided by tasks

X = mat_contents['X_train']
Y = mat_contents['Y_train']
Xt = mat_contents['X_test']
Yt = mat_contents['Y_test']

# ResNet18 pretrained network

resnet18 = models.resnet18(pretrained=True)
features_resnet18 = nn.Sequential(*(list(resnet18.children())[:-1]))
features_resnet18.eval()

transform = transforms.Compose(                    
    [transforms.Resize(256),                             
      transforms.CenterCrop(224),                      
      transforms.ToTensor(),                              
      transforms.Normalize(mean=[0.485, 0.456, 0.406],     
                          std=[0.229, 0.224, 0.225])])  

X_train = []
Y_train = []
X_test = []
Y_test = []

for i in range(0, len(X[0, :])):
    X1 = X[0, i]
    Y1 = Y[0, i]
    X_features = []
    Y_features = []
    for j in range(0, len(X1[0, 0, 0, :])):
        X2 = X1[:, :, :, j]
        Y2 = Y1[0, j]
        PIL_image = Image.fromarray(X2.astype('uint8'), 'RGB')
        img_t = transform(PIL_image)
        batch_t = torch.unsqueeze(img_t, 0)
        X_features.append(features_resnet18(batch_t).detach().numpy().flatten())
        Y_features.append(Y2)
    X_train.append(np.array(X_features))
    Y_train.append(np.array(Y_features))
    Xt1 = Xt[0, i]
    Yt1 = Yt[0, i]
    Xt_features = []
    Yt_features = []
    for j in range(0, len(Xt1[0, 0, 0, :])):
        Xt2 = Xt1[:, :, :, j]
        Yt2 = Yt1[0, j]
        PIL_image = Image.fromarray(Xt2.astype('uint8'), 'RGB')
        img_t = transform(PIL_image)
        batch_t = torch.unsqueeze(img_t, 0)
        Xt_features.append(features_resnet18(batch_t).detach().numpy().flatten())
        Yt_features.append(Yt2)
    X_test.append(np.array(Xt_features))
    Y_test.append(np.array(Yt_features))

mdic = {"X_train": np.array(X_train), "Y_train": np.array(Y_train), "X_test": np.array(X_test), "Y_test": np.array(Y_test)}
savemat("data.mat", mdic)