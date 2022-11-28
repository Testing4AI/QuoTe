import os
import sys
sys.path.append("../")
sys.path.append("../../")

import numpy as np
import imageio
import cv2

import torch
import model
from PIL import Image

from utils import util

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
mean = np.array([129.1863, 104.7624, 93.5940]) #RGB

# protected = "gender"
protected = "race"

race_transformers = {"Caucasian": ["AfricantoCaucasian", "CaucasiantoAsian", "CaucasiantoIndian"],
                     "African": ["AfricantoCaucasian", "AfricantoAsian", "AfricantoIndian"],
                     "Asian": ["CaucasiantoAsian", "AfricantoAsian", "AsiantoIndian"],
                     "Indian": ["CaucasiantoIndian", "AfricantoIndian", "AsiantoIndian"]}
race_directions = {"Caucasian": ["BtoA", "AtoB", "AtoB"], "African": ["AtoB", "AtoB", "AtoB"],
                  "Asian": ["BtoA", "BtoA", "AtoB"], "Indian": ["BtoA", "BtoA", "BtoA"]}
gender_transformers = {"Male": ["MaletoFemale"], "Female": ["MaletoFemale"]}
gender_directions ={"Male": ["AtoB"], "Female": ["BtoA"]}
size = (224, 224) #VGGFace
seed = 3000 # seed number of raw images

if protected == "race":
    transformers, directions = race_transformers, race_directions
elif protected == "gender":
    transformers, directions = gender_transformers, gender_directions

# initialize hyper parameter
w = 224
h = 224

info = np.loadtxt('../datasets/fairface/fairface_label_test.csv', dtype=np.str, delimiter=',')[1:]
[names, ages, genders, races] = [info[:, 0], info[:, 1], info[:, 2], info[:, 3]]

from options.test_options import TestOptions
opt = TestOptions().parse()

store_path = "../datasets/fairface/generated_" + protected + "/"
if not os.path.exists(store_path):
    os.makedirs(store_path)

def generate(x, source, target, img_name, file_path):
    image_name = '%s_%s.png' % (img_name.split(".")[0], target)
    save_path = os.path.join(file_path, image_name)
    if not os.path.exists(save_path):
        candidate = transformers[source]
        if source + "to" + target in candidate:
            idx = candidate.index(source + "to" + target)
            opt.direction = "AtoB"
        elif target + "to" + source in candidate:
            idx = candidate.index(target + "to" + source)
            opt.direction = "BtoA"

        opt.name = candidate[idx]

        transformer = ftc_model.create_model(opt)
        transformer.setup(opt)
        input_nc = opt.output_nc if opt.direction == 'BtoA' else opt.input_nc
        x_input = get_transform(opt, grayscale=(input_nc == 1))(Image.fromarray(x))

        input = {'A': x_input.unsqueeze(0), 'A_paths': img_name}

        transformer.eval()
        transformer.set_input(input)
        transformer.test()

        visuals = transformer.get_current_visuals()  # get image results
        im = util.tensor2im(visuals['fake'])
        util.save_image_size(im, save_path, size)
    x_new  = imageio.imread(save_path)
    return x_new

def check_attribute(a_name):
    if a_name == "White":
        return "Caucasian"
    if a_name == "Black":
        return "African"
    if a_name == "East Asian":
        return "Asian"
    if a_name == "M":
        return "Male"
    if a_name == "F":
        return "Female"
    return a_name

def construct(batch, e_batch, n_batch, file_path):
    array = []
    for num in range(len(batch)):
        x = batch[num]
        img_name = n_batch[num]
        array.append(x.copy())
        e = e_batch[num]
        e = check_attribute(e)
        if protected == "race" and e in ["Caucasian", "African", "Asian", "Indian"]:
            for ev in ["Caucasian", "African", "Asian", "Indian"]:
                if ev != e:
                    g_x = generate(x.copy(), e, ev, img_name, file_path)
                    array.append(g_x.copy())
        elif protected == "gender" and e in ["Male","Female"]:
            for ev in ["Male","Female"]:
                if ev != e:
                    g_x = generate(x.copy(), e, ev, img_name, file_path)
                    array.append(g_x.copy())
    return np.array(array)

path = '../datasets/fairface/test/'
for i in range(len(names)):
    file = names[i]
    if protected == "race":
        e_batch = races[i]
    elif protected == "gender":
        e_batch = genders[i]

    img = imageio.imread(path + file, pilmode='RGB').copy()

    n_batch = file
    x_array = construct(np.array([img]), np.array([e_batch]), np.array([n_batch]),
                        store_path)  # num_domain * num_sample



