import sys
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
import random
from model import resnet50

mean = np.array([129.1863, 104.7624, 93.5940]) #RGB
size = (224, 224) #VGGFace
seed = 3

age_bins = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]


gpu_ids = ['2','3']
cuda = "cuda:2" 
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

# gpu_ids = ['0','1']
# cuda = "cuda:" + gpu_ids[0]
# device = torch.device(cuda if torch.cuda.is_available() else "cpu")

# model = resnet50(num_classes=9, include_top=True)
# model = nn.DataParallel(model, device_ids=[int(id) for id in gpu_ids])
# model.to(device)

# model.load_state_dict(torch.load("./models_age/resnet50_60.pth"))
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
# optimizer.load_state_dict(torch.load("./models_age/resnet50_60_optim.pth"))
# model.train()

store_path = "./retrain_models_resnet/"

criterion = nn.CrossEntropyLoss()

with np.load("./tests_resnet.npz") as f:
    all_names = f['all_names']
    all_fols = f['all_fols']
    all_ginis = f['all_ginis']
    all_ages = f['all_ages']

with np.load("./sub_train.npz") as f:
    sub_names = f['sub_names']
    sub_ages = f['sub_ages']

    

def select(values, n, s='best', k=100):
    """
    n: the number of selected test cases. 
    s: strategy, ['best', 'random', 'kmst', 'gini']
    k: for KM-ST, the number of ranges. 
    """
    ranks = np.argsort(values) 
    
    if s == 'best':
        h = n//2
        return np.concatenate((ranks[:h],ranks[-h:]))
        
    elif s == 'r':
        return np.array(random.sample(list(ranks),n)) 
    
    elif s == 'kmst':
        fol_max = values.max()
        th = fol_max / k
        section_nums = n // k
        indexes = []
        for i in range(k):
            section_indexes = np.intersect1d(np.where(values<th*(i+1)), np.where(values>=th*i))
            if section_nums < len(section_indexes):
                index = random.sample(list(section_indexes), section_nums)
                indexes.append(index)
            else: 
                indexes.append(section_indexes)
                index = random.sample(list(ranks), section_nums-len(section_indexes))
                indexes.append(index)
        return np.concatenate(np.array(indexes))

    # This is for gini strategy. There is little difference from DeepGini paper. See function ginis() in metrics.py 
    else: 
        return ranks[:n]   
    


# 1%
for stra in ['best']:
    model = resnet50(num_classes=9, include_top=True)
    model = nn.DataParallel(model, device_ids=[int(id) for id in gpu_ids])
    model.to(device)
    model.load_state_dict(torch.load("./models_age/resnet50_60.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    optimizer.load_state_dict(torch.load("./models_age/resnet50_60_optim.pth"))
    model.train()
    
    if stra == 'g':
        select_idx = select(all_ginis, 500, s=stra)    
    else:
        select_idx = select(all_fols, 500, s=stra)
        
    Xnames = np.concatenate((sub_names, all_names[select_idx]))
    Xages = np.concatenate((sub_ages, all_ages[select_idx]))
    
    indices = np.array(range(len(Xnames)))
    np.random.shuffle(indices)
    nameX = Xnames[indices]
    ageLabel = Xages[indices]
    
    batch_size = 128
    for epoch in range(10):
        batch = math.ceil(len(nameX) / batch_size)
        for bt in range(batch):
            start_idx = bt * batch_size
            end_idx = min((bt + 1) * batch_size, len(nameX))
            num = end_idx - start_idx
            name_batch = nameX[start_idx:end_idx]

            x_batch = []
            for name in name_batch:
                x_batch.append(imageio.imread(name, pilmode='RGB').copy())
            x_batch = np.array(x_batch)

            y_batch = ageLabel[start_idx:end_idx]

            x_b = torch.Tensor(x_batch).to(device).permute(0, 3, 1, 2).view(len(x_batch), 3, 224, 224)
            x_b -= torch.Tensor(mean).to(device).view(1, 3, 1, 1)

            optimizer.zero_grad()
            output = model(x_b)
            loss = criterion(output, torch.tensor(y_batch).to(device))

            loss.backward()
            optimizer.step()

        if epoch % 3 == 0:
            torch.save(model.state_dict(), store_path + "resnet50_500_" + stra + "_" + str(epoch) + ".pth")
