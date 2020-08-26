from __future__ import print_function, division
import os
import random
import torch
import pandas as pd
from random import sample
import collections
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torchvision.transforms.functional as TF
#from torchsample.transforms import RandomRotate
import torchvision.transforms as transforms
from torch.utils import data
#import cv2
import warnings

warnings.filterwarnings("ignore")

## Dataset definition file
## Dataset at : https://drive.google.com/drive/folders/1mwZJ06zerm9-z7Q606xPip1SiTnxduqY?usp=sharing

class TrajDataset(Dataset):
    ## Initialization##
    def __init__(self, img_file, transform, split):
        self.dir_img = '/path/to/datasets/escape_data/autoencoder_data/train_autoencoder' ## path to all dataset images
        self.img_lst = img_file  ## train/validation/test containing the names of training images and testing images
        self.transform = transform
        self.files = collections.defaultdict(list)
        self.split = split

        for images in self.img_lst:
            image_paths  = os.path.join(self.dir_img, images)
            label       = (images.split(".")[0]).split("_")[1]
            self.files[self.split].append({
                "image" : image_paths,
                "label" : label
            })


    def __getitem__(self, index):

        ## read images and masks. some masks are 3 dimensional , hence converting to gray-scale
        datafiles = self.files[self.split][index]
        img_file = datafiles["image"]
        label = datafiles["label"]

        image = Image.open(img_file).convert("RGB")


        if self.transform is not None:
            images = self.transform(image, self.split)

        #print(image.size, mask.size)
        return images, label

    ## Total Length of Dataset ##
    def __len__(self):
        return len(self.files[self.split])


def transform(image, split):
    ## Random crop - cropping images randomly into 256 by 256 by 3##
    if split =='train':
        resize = transforms.Resize(size= (256,256), interpolation=Image.BILINEAR)
        image = resize(image)
        if random.random() > 0.5:
            image = TF.hflip(image)

        ## Random rotate ##
        if random.random() > 0.5:
            rt_angles = [30, 45, 90]  ##Added 30, 45 degrees
            rt_indx = np.random.randint(2, dtype='int')
            angle = rt_angles[rt_indx]
            image = TF.rotate(image, angle, resample=False, expand=False, center=None)

        image = TF.to_tensor(image)
    else:
        resize = transforms.Resize(size= (256,256), interpolation=Image.BILINEAR)
        image = resize(image)
        image = TF.to_tensor(image)

    return image



if __name__ == '__main__': ##for debugging


    img_file = 'train_list.txt'
    base_directory = '/path/to/datasets/escape_data/autoencoder_data'
    img_list = [line.rstrip('\n') for line in open(os.path.join(base_directory, img_file))]
    print("length of training/validation dataset: ", len(img_list))
    img_shuffled = sample(img_list, len(img_list))
    train_ind = int(0.9 * len(img_shuffled))
    train_lst = img_shuffled[:train_ind]
    print("length of training dataset: ", len(train_lst))
    print("First 10 training images", train_lst[:10])
    val_lst = img_shuffled[train_ind:]

    dst = TrajDataset(train_lst, transform, split='train')
    trainloader = DataLoader(dst, batch_size=10)

    print("Length of trainloader:", len(trainloader))


    for i, data in enumerate(trainloader):
        print("---------------------------------")
        print (i)
        imgs, labels = data

        print("---------------------------------")
        print("Images type, Images length, 0th Image shape: ", type(imgs), "---", len(imgs),"---", imgs[0].shape)
        print("Labels Type, Labels length: ", type(labels), "---", len(labels))
        if i == 90:
            images = imgs
            print(len(images), images.shape)
            for j in range(len(images)):
                z = images[j]
                ax = plt.subplot(4, 4, (j + 1))
                plt.imshow(z.permute(1, 2, 0))
        plt.savefig("image_autoencoder_show.png")





