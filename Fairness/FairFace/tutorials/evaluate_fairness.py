# -*- coding: utf-8 -*-
import sys
sys.path.append('../../')
sys.path.append('../')
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch import nn
import argparse
import imageio
from PIL import Image
import os
import math

import model
from utils import util
from utils.util import *
from utils.model_utils import load_model_resnet

mean = np.array([129.1863, 104.7624, 93.5940]) #RGB
size = (224, 224) #VGGFace
seed = 3

parser = argparse.ArgumentParser()

protected = "race"
pa_values = 4

store_path = "../datasets/fairface/generated_" + protected + "/"

def construct(batch, e_batch, n_batch, file_path):
    array = []
    for num in range(len(batch)):
        x = batch[num]
        img_name = n_batch[num]
        array.append(x.copy())
        e = e_batch[num]
        e = check_attribute(e)
        for ev in ["Caucasian", "African", "Asian", "Indian"]:
            if ev != e:
                image_name = '%s_%s.png' % (img_name.split(".")[0], ev)
                save_path = os.path.join(file_path, image_name)
                g_x = imageio.imread(save_path)
                array.append(g_x.copy())
    return np.array(array)


if __name__ == "__main__":
    parser.add_argument('--gpu', type=str, default="0", help='')
    parser.add_argument('--model_path', type=str, default="./trained_models/resnet50_99.pth", help='')
    args = parser.parse_args()

    model_path = args.model_path

    gpu_ids = args.gpu.split(',')
    cuda = "cuda:" + gpu_ids[0]
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")

    model = load_model_resnet(gpu_ids,model_path,"age")

    info = np.loadtxt('../datasets/fairface/fairface_label_test.csv', dtype=np.str, delimiter=',')[1:]
    [names, ages, genders, races] = [info[:, 0], info[:, 1], info[:, 2], info[:, 3]]

    acc_gender = {"m_correct":0, "m_total":0, "f_correct":0, "f_total":0, "mf_correct":0, "mf_total":0}
    acc_ethnic = {"w_correct": 0, "w_total": 0, "b_correct": 0, "b_total": 0, "a_correct": 0, "a_total": 0, "i_correct": 0, "i_total": 0, "wbai_correct": 0, "wbai_total": 0}

    path = '../datasets/fairface/test/'
    files = sorted(os.listdir(path))

    fair_dict = {}
    for file in files:
        l = np.where(names == file)[0][0]
        img = imageio.imread(path + file, pilmode='RGB')
        x = np.array([img.copy()])
        # y = np.array([ages[l]])

        e_batch = np.array([races[l]])
        n_batch = [file]
        if races[l] in ["White", "Black", "East Asian", "Indian"]:
            x_array = construct(x, e_batch, n_batch, store_path)  # num_domain * num_sample

            x_b = torch.Tensor(x_array).to(device).permute(0, 3, 1, 2).view(len(x_array), 3, 224, 224)
            x_b -= torch.Tensor(mean).to(device).view(1, 3, 1, 1)
            _, pred = F.softmax(model(x_b), -1).max(-1)
            # print(pred.data.numpy(),y, pred.data.numpy()==y)

            preds = pred.data.cpu().numpy()
            if races[l] == "White":
                acc_ethnic["w_total"] = acc_ethnic["w_total"] + 1
                if preds.sum() == preds[0] * pa_values:
                    acc_ethnic["w_correct"] = acc_ethnic["w_correct"] + 1
                same = [-1, 0, 0, 0]
                for num in [1,2,3]:
                    if preds[num] == preds[0]:
                        same[num] = 1
                fair_dict[file] = same
            elif races[l] == "Black":
                acc_ethnic["b_total"] = acc_ethnic["b_total"] + 1
                if preds.sum() == preds[0] * pa_values:
                    acc_ethnic["b_correct"] = acc_ethnic["b_correct"] + 1
                same = [0, -1, 0, 0]
                for num in [0,2,3]:
                    if preds[num] == preds[1]:
                        same[num] = 1
                fair_dict[file] = same
            elif races[l] == "East Asian":
                acc_ethnic["a_total"] = acc_ethnic["a_total"] + 1
                if preds.sum() == preds[0] * pa_values:
                    acc_ethnic["a_correct"] = acc_ethnic["a_correct"] + 1
                same = [0, 0, -1, 0]
                for num in [0,1,3]:
                    if preds[num] == preds[2]:
                        same[num] = 1
                fair_dict[file] = same
            elif races[l] == "Indian":
                acc_ethnic["i_total"] = acc_ethnic["i_total"] + 1
                if preds.sum() == preds[0] * pa_values:
                    acc_ethnic["i_correct"] = acc_ethnic["i_correct"] + 1
                same = [0, 0, 0, -1]
                for num in [0,1,2]:
                    if preds[num] == preds[3]:
                        same[num] = 1
                fair_dict[file] = same
            else:
                acc_ethnic["wbai_total"] = acc_ethnic["wbai_total"] + 1
                if preds.sum() == preds[0] * pa_values:
                    acc_ethnic["wbai_correct"] = acc_ethnic["wbai_correct"] + 1
    # np.save("./fair_race_dict.npy",fair_dict)
    print(acc_ethnic)
