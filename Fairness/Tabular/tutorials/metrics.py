import sys
import torch
import numpy as np
from torch import nn
import imageio
import os



def fol(model, all_names, all_labels):
    fols = []
    criterion = nn.CrossEntropyLoss()
    for i in range(1000):
        gname = all_names[i]
        age = all_labels[i]

        xs = []
        ys = []
        img = imageio.imread(gname, pilmode='RGB').copy()
        xs.append(img.copy())
        ys.append(age)
        xs = np.array(xs)
        ys = np.array(ys)

        x = torch.Tensor(xs).to(device).permute(0, 3, 1, 2).view(len(xs), 3, 224, 224)
        x.requires_grad = True
        x_b = x - torch.Tensor(mean).to(device).view(1, 3, 1, 1)

        output = model(x_b)
        loss = criterion(output, torch.tensor(ys).to(device))
        loss.backward(retain_graph=True)

        gd = x.grad.cpu().numpy()
        fol = np.linalg.norm(gd.reshape(1,-1), ord=2, axis=1)[0]

        fols.append(fol)
    
    return np.array(fols) 



def zol(model, all_names, all_labels):
    zols = []
    criterion = nn.CrossEntropyLoss()
    for i in range(1000):
        gname = all_names[i]
        age = all_labels[i]

        xs = []
        ys = []
        img = imageio.imread(gname, pilmode='RGB').copy()
        xs.append(img.copy())
        ys.append(age)
        xs = np.array(xs)
        ys = np.array(ys)

        x = torch.Tensor(xs).to(device).permute(0, 3, 1, 2).view(len(xs), 3, 224, 224)
        x.requires_grad = True
        x_b = x - torch.Tensor(mean).to(device).view(1, 3, 1, 1)

        output = model(x_b)
        loss = criterion(output, torch.tensor(ys).to(device))

        zols.append(loss.numpy()[0])
    
    return np.array(zols) 