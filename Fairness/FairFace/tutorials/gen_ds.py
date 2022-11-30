# distance diversity: the L2 distance (or consine similarity)
# between GA->B(xA) and xA

import os
import sys
sys.path.append("../")
sys.path.append("../../")

import numpy as np
import imageio
import math
import cv2
from skimage.restoration import denoise_tv_bregman
import torch
import torch.nn.functional as F
import model
from PIL import Image
import random
from torch import nn
import argparse

from utils.model_utils import load_model_resnet
from utils.util import *
from utils import config

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="fairface", help='')
parser.add_argument('--sens', type=str, default="race", help='')
parser.add_argument('--model_path', type=str, default="./trained_models/resnet50_99.pth", help='')
parser.add_argument('--gpu', type=str, default="0", help='')
parser.add_argument('--name', type=str, default="age", help='None, gender, race')
parser.add_argument('--max_iter', type=int, default=10, help='')
parser.add_argument('--step_size', type=int, default=5, help='')

args = parser.parse_args()

gpu_ids, device = gpu_parse(args)

model = load_model_resnet(gpu_ids, args.model_path, args.name)

protected = args.sens

if protected == "race":
    pa_values = 4
elif protected == "gender":
    pa_values = 2

mean = config.mean



# Gradient
info = np.loadtxt('../datasets/fairface/fairface_label_test.csv', dtype=np.str, delimiter=',')[1:]
[names, ages, genders, races] = [info[:, 0], info[:, 1], info[:, 2], info[:, 3]]

test_path = config.test_path + args.dataset + "/test/"
# test_path = "../../datasets/" + args.dataset + "/test/"
files = sorted(os.listdir(test_path))

generated_path = config.test_path + args.dataset + "/generated_" + protected + "/"
# generated_path = "../../datasets/" + args.dataset + "/generated_" + protected + "/"

path = config.test_path + args.dataset + "/adf_pair_" + protected + "_"+str(args.step_size)+"/"
# path = "../../datasets/" + args.dataset + "/adf_pair_" + protected + "_"+str(args.step_size)+"/"
create_dir(path)

race_array = ["Caucasian", "African", "Asian", "Indian"]
succ = 0

for file in files:
    l = np.where(names == file)[0][0]
    e = check_attribute(races[l])

    if protected == "race" and e in ["Caucasian", "African", "Asian", "Indian"]:
        for ev in ["Caucasian", "African", "Asian", "Indian"]:
            if ev != e:
                xs = []
                ys = []
                f = '%s_%s.png' % (file[:-4], ev)
                img = imageio.imread(test_path + file, pilmode='RGB')
                g_img = imageio.imread(generated_path + f, pilmode='RGB')
                xs.append(img.copy())
                xs.append(g_img.copy())
                xs = np.array(xs)

                for iter in range(args.max_iter + 1):
                    x = torch.Tensor(xs).to(device).permute(0, 3, 1, 2).view(len(xs), 3, 224, 224)
                    x.requires_grad = True
                    x_b = x - torch.Tensor(mean).to(device).view(1, 3, 1, 1)

                    preds_torch = F.softmax(model(x_b), -1) # N * C
                    preds = preds_torch.data.cpu().numpy()
                    labels = np.argmax(preds,axis=-1)

                    if labels.sum() != labels[0] * len(xs):
                        print(file, iter)
                        if iter != 0:
                            succ += 1
                            for xImg in xs[1:]:
                                f = '%s_2%s.png' % (file[:-4], ev)
                                cv2.imwrite(path + f, cv2.cvtColor(xs[0], cv2.COLOR_RGB2BGR))
                                f = '%s_%s.png' % (file[:-4], ev)
                                cv2.imwrite(path + f, cv2.cvtColor(xImg, cv2.COLOR_RGB2BGR))
                        break

                    if iter == args.max_iter:
                        print(file, iter)
                        break

                    # select most difference
                    probs = np.max(preds,axis=-1)
                    sIdx = np.argmax(abs(probs - probs[0]))

                    # compute gradient
                    preds_torch, _ = preds_torch.max(-1)
                    preds_torch[0].backward(retain_graph=True)
                    preds_torch[sIdx].backward(retain_graph=True)
                    gd = x.grad.cpu().numpy()

                    # select attribute and direction
                    sign1 = np.sign(gd[0])
                    sign2 = np.sign(gd[sIdx])
                    direction = sign1 * (sign1 == sign2)

                    # apply perturbation
                    xs = xs - direction.transpose((1,2,0)) * args.step_size

                    # clip
                    xs = np.clip(xs, 0, 255)

print(succ)



# Random 
info = np.loadtxt('../datasets/fairface/fairface_label_test.csv', dtype=np.str, delimiter=',')[1:]
[names, ages, genders, races] = [info[:, 0], info[:, 1], info[:, 2], info[:, 3]]

test_path = config.test_path + args.dataset + "/test/"
files = sorted(os.listdir(test_path))

generated_path = config.test_path + args.dataset + "/generated_" + protected + "/"

path = config.test_path + args.dataset + "/random_pair_" + protected + "_"+str(args.step_size)+"/"
create_dir(path)

race_array = ["Caucasian", "African", "Asian", "Indian"]
succ = 0

for file in files:
    l = np.where(names == file)[0][0]
    e = check_attribute(races[l])

    if protected == "race" and e in ["Caucasian", "African", "Asian", "Indian"]:
        for ev in ["Caucasian", "African", "Asian", "Indian"]:
            if ev != e:
                xs = []
                ys = []
                f = '%s_%s.png' % (file[:-4], ev)
                img = imageio.imread(test_path + file, pilmode='RGB')
                g_img = imageio.imread(generated_path + f, pilmode='RGB')
                xs.append(img.copy())
                xs.append(g_img.copy())
                xs = np.array(xs)

                for iter in range(args.max_iter + 1):
                    x = torch.Tensor(xs).to(device).permute(0, 3, 1, 2).view(len(xs), 3, 224, 224)
                    x.requires_grad = True
                    x_b = x - torch.Tensor(mean).to(device).view(1, 3, 1, 1)

                    preds_torch = F.softmax(model(x_b), -1) # N * C
                    preds = preds_torch.data.cpu().numpy()
                    labels = np.argmax(preds,axis=-1)

                    if labels.sum() != labels[0] * len(xs):
                        print(file, iter)
                        if iter != 0:
                            succ += 1
                            for xImg in xs[1:]:
                                f = '%s_2%s.png' % (file[:-4], ev)
                                cv2.imwrite(path + f, cv2.cvtColor(xs[0], cv2.COLOR_RGB2BGR))
                                f = '%s_%s.png' % (file[:-4], ev)
                                cv2.imwrite(path + f, cv2.cvtColor(xImg, cv2.COLOR_RGB2BGR))
                        break

                    if iter == args.max_iter:
                        print(file, iter)
                        break

                    # select attribute and direction
                    direction = np.random.randint(-1, 1, (224, 224, 3)).astype(xs.dtype)

                    # apply perturbation
                    xs = xs - direction * args.step_size

                    # clip
                    xs = np.clip(xs, 0, 255)

print(succ)


# Gausian
info = np.loadtxt('../datasets/fairface/fairface_label_test.csv', dtype=np.str, delimiter=',')[1:]
[names, ages, genders, races] = [info[:, 0], info[:, 1], info[:, 2], info[:, 3]]

test_path = config.test_path + args.dataset + "/test/"
files = sorted(os.listdir(test_path))

generated_path = config.test_path + args.dataset + "/generated_" + protected + "/"

path = config.test_path + args.dataset + "/gaussian_pair_" + protected + "_"+str(args.sigma)+"/"
create_dir(path)

race_array = ["Caucasian", "African", "Asian", "Indian"]
succ = 0

for file in files:
    l = np.where(names == file)[0][0]
    e = check_attribute(races[l])

    if protected == "race" and e in ["Caucasian", "African", "Asian", "Indian"]:
        for ev in ["Caucasian", "African", "Asian", "Indian"]:
            if ev != e:
                xs = []
                ys = []
                f = '%s_%s.png' % (file[:-4], ev)
                img = imageio.imread(test_path + file, pilmode='RGB')
                g_img = imageio.imread(generated_path + f, pilmode='RGB')
                xs.append(img.copy())
                xs.append(g_img.copy())
                xs = np.array(xs)

                for iter in range(args.max_iter + 1):
                    x = torch.Tensor(xs).to(device).permute(0, 3, 1, 2).view(len(xs), 3, 224, 224)
                    x.requires_grad = True
                    x_b = x - torch.Tensor(mean).to(device).view(1, 3, 1, 1)

                    preds_torch = F.softmax(model(x_b), -1) # N * C
                    preds = preds_torch.data.cpu().numpy()
                    labels = np.argmax(preds,axis=-1)

                    if labels.sum() != labels[0] * len(xs):
                        print(file, iter)
                        if iter != 0:
                            succ += 1
                            for xImg in xs[1:]:
                                f = '%s_2%s.png' % (file[:-4], ev)
                                cv2.imwrite(path + f, cv2.cvtColor(xs[0], cv2.COLOR_RGB2BGR))
                                f = '%s_%s.png' % (file[:-4], ev)
                                cv2.imwrite(path + f, cv2.cvtColor(xImg, cv2.COLOR_RGB2BGR))
                        break

                    if iter == args.max_iter:
                        print(file, iter)
                        break

                    # gaussian noise
                    gaussian = np.random.randn(224, 224, 3).astype(xs.dtype) * args.sigma

                    # apply perturbation
                    xs = xs - gaussian

                    # clip
                    xs = np.clip(xs, 0, 255)

print(succ)













