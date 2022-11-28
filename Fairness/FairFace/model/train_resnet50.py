import sys
# sys.path.append('../../')
# sys.path.append('../')
import torch
import cv2
import numpy as np
import random
import torch.nn.functional as F
from torch import nn
import argparse
import imageio
from PIL import Image
import os
import math
from models import resnet50


mean = np.array([129.1863, 104.7624, 93.5940]) #RGB
size = (224, 224) #VGGFace
seed = 3
age_bins = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]

gpu_ids = ['0','1','2','3']
cuda = "cuda:" + gpu_ids[0]
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

model = resnet50(num_classes=9, include_top=True)
if len(gpu_ids) > 1:
    model = nn.DataParallel(model, device_ids=[int(id) for id in gpu_ids])
model.to(device)
    
    
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        
model.apply(weight_init)
model.train()

info = np.loadtxt('./datasets/fairface_label_train.csv', dtype=np.str, delimiter=',')[1:]
[names, ages, genders, races]=[info[:,0], info[:,1], info[:,2], info[:,3]]
train_path = './datasets/'

indices = np.array(range(len(names)))
np.random.shuffle(indices)
names = names[indices]
ages = ages[indices]
for i in range(len(ages)):
    ages[i] = age_bins.index(ages[i])
ages = ages.astype(int)
store_path = "./models_age/"


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

batch_size = 128
batch = math.ceil(len(names) / batch_size)
for epoch in range(80):
    for bt in range(batch):
        start_idx = bt * batch_size
        end_idx = min((bt + 1) * batch_size, len(names))
        num = end_idx - start_idx
        name_batch = names[start_idx:end_idx]

        x_batch = []
        for name in name_batch:
            x_batch.append(imageio.imread(train_path+name, pilmode='RGB').copy())
        x_batch = np.array(x_batch)

        y_batch = ages[start_idx:end_idx]

        x_b = torch.Tensor(x_batch).to(device).permute(0, 3, 1, 2).view(len(x_batch), 3, 224, 224)
        x_b -= torch.Tensor(mean).to(device).view(1, 3, 1, 1)

        optimizer.zero_grad()
        output = model(x_b)
        loss = criterion(output, torch.tensor(y_batch).to(device))

#         print(loss.data)
        loss.backward()
        optimizer.step()
        
    if epoch % 10 == 0 or epoch == 100 - 1:
        torch.save(model.state_dict(), store_path + "resnet50_" + str(epoch) + ".pth")
        torch.save(optimizer.state_dict(), store_path + "resnet50_" + str(epoch) + "_optim.pth")
        

        

