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

sys.path.append("../")
sys.path.append("../../")
from models import vgg16

gpu_ids = ['2','3']
cuda = "cuda:" + gpu_ids[0]
device = torch.device(cuda if torch.cuda.is_available() else "cpu")


mean = np.array([129.1863, 104.7624, 93.5940]) #RGB
size = (224, 224) #VGGFace
seed = 3

age_bins = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]

def eval(model, X, Y):
    accuracy = 0
    batch_size = 128
    batch = math.ceil(len(X) / batch_size)
    for bt in range(batch):
        start_idx = bt * batch_size
        end_idx = min((bt + 1) * batch_size, len(X))
        num = end_idx - start_idx
        name_batch = X[start_idx:end_idx]

        x_batch = []
        for name in name_batch:
            x_batch.append(imageio.imread(name, pilmode='RGB').copy())
        x_batch = np.array(x_batch)

        y_batch = Y[start_idx:end_idx]

        x_b = torch.Tensor(x_batch).to(device).permute(0, 3, 1, 2).view(len(x_batch), 3, 224, 224)
        x_b -= torch.Tensor(mean).to(device).view(1, 3, 1, 1)
        with torch.no_grad():
            _, preds = F.softmax(model(x_b), -1).max(-1)
        predsnp = preds.data.cpu().numpy()
        for idx in range(len(predsnp)):
            if predsnp[idx] == y_batch[idx]:
                accuracy += 1

    print('ACC: ', accuracy/len(X))   


with np.load("./tests_vgg.npz") as f:
    all_names = f['all_names']
    all_fols = f['all_fols']
    all_ginis = f['all_ginis']
    all_ages = f['all_ages']


model_path = "./retrain_models_vgg/"
model_names = sorted(os.listdir(model_path))


for name in model_names:
    print(name, ":", end=' ')
    model_path = "./retrain_models_vgg/" + name
    model = vgg16()
    model = nn.DataParallel(model, device_ids=[int(id) for id in gpu_ids])
    model.load_state_dict(torch.load(model_path)) 
    model.to(device)
    eval(model, all_names, all_ages)
