import os
from deepfait_data.base_dataset import BaseDataset, get_transform
from deepfait_data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random

class CelebADataset(BaseDataset):
    """
    This dataset class can load Fairface dataset.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.meta_file = opt.meta_file
        np.random.seed(2021)

        # open data
        with open(self.meta_file) as f:
            data = np.loadtxt(f, str, delimiter=",", skiprows=1)

        # extract data
        self.img_path = data[:, 0]
        self.gender = np.array([1 if g == "1" else 0 for g in data[:, 21]])

        # match pairs with same ethnic
        self.index_A = np.where(self.gender == 1)[0]
        self.index_B = np.where(self.gender == 0)[0]
        self.A_paths = self.img_path[self.index_A]
        self.B_paths = self.img_path[self.index_B]

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(os.path.join(self.opt.dataroot, A_path)).convert('RGB')
        B_img = Image.open(os.path.join(self.opt.dataroot, B_path)).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)