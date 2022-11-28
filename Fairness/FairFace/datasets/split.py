import os
import numpy as np
import imageio
from PIL import Image

store_path = "./test/"
if not os.path.exists(store_path):
    os.makedirs(store_path)

path = "./"
fileNames = sorted(os.listdir(path))

for fileName in fileNames:
    if not os.path.exists(store_path + fileName):
        os.makedirs(store_path + fileName)
    files = sorted(os.listdir(path + fileName))
    testFiles = np.random.choice(files,20,replace=False)
    for file in testFiles:
        try:
            img = imageio.imread(path + fileName +'/' + file)
            image = Image.fromarray(img)
            image.save(store_path + fileName +'/' + file)
        except Exception as e:
            print(file)

