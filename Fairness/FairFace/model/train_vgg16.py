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


class VGG(nn.Module):

    def __init__(self, features, num_classes=9, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model

def vgg13(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model



gpu_ids = ['2','3']
cuda = "cuda:2" 
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

model = vgg16()
if len(gpu_ids) > 1:
    model = nn.DataParallel(model, device_ids=[int(id) for id in gpu_ids])
model.to(device)
    
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
for epoch in range(71):
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
        
    if epoch % 10 == 0:
        torch.save(model.state_dict(), store_path + "vgg16_" + str(epoch) + ".pth")
        torch.save(optimizer.state_dict(), store_path + "vgg16_" + str(epoch) + "_optim.pth")
        

