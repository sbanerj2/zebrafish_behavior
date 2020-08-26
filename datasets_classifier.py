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
## Code for prepping the trajectory images for training classifier 

class TrajectoryDataset(Dataset):
    ## Initialization##
    def __init__(self, img_file='train_cls_list.txt', transform=None, splits='train'):
        self.dir_img = '/path/to/escape_data/classifier_data/classifier_data/train_classifier' ## path to all dataset images
        self.img_lst = img_file  ## train/validation/test containing the names of training images and testing images
        self.transform = transform
        self.files = collections.defaultdict(list)
        self.splits = splits

        for images in self.img_lst:
            image_paths  = os.path.join(self.dir_img, images)
            label       = (images.split(".")[0]).split("-")[-1] #(names.split(".")[0]).split("-")[-1]
            self.files[self.splits].append({
                "image" : image_paths,
                "label" : label
            })


    def __getitem__(self, index):

        ## read images and labels
        datafiles = self.files[self.splits][index]
        img_file = datafiles["image"]
        label = datafiles["label"]

        image = Image.open(img_file).convert("RGB")
        
        if 'yes' in label:
        	labels = 1
        else:
        	labels = 0


        if self.transform is not None:
            images = self.transform(image, self.splits)

        return images, labels

    ## Total Length of Dataset ##
    def __len__(self):
        return len(self.files[self.splits])


def transform(image, splits):
    ## Random crop - cropping images randomly into 256 by 256 by 3##
    if splits =='train':
        ## Random crop - cropping images randomly into 256 by 256 by 3##
        ## https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/6
        ## Random horizontal flipping ##
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



if __name__ == '__main__': ##for debug


    img_file = 'train_cls_list.txt'
    base_directory_path = '/path/to/escape_data/classifier_data/classifier_data/'
    img_list = [line.rstrip('\n') for line in open(os.path.join(base_directory_path, img_file))]
    print("length of training/validation dataset: ", len(img_list))
    img_shuffled = sample(img_list, len(img_list))
    train_ind = int(0.9 * len(img_shuffled))
    train_lst = img_shuffled[:train_ind]
    print("length of training dataset: ", len(train_lst))
    print("First 10 training images", train_lst[:10])
    val_lst = img_shuffled[train_ind:]

    dst = TrajectoryDataset(train_lst, transform, splits='train')
    trainloader = DataLoader(dst, batch_size=16)

    print("Length of trainloader:", len(trainloader))


    for i, data in enumerate(trainloader):
        print("---------------------------------")
        ##https://github.com/ycszen/pytorch-segmentation/blob/master/datasets.py
        print (i)
        imgs, labels = data

        print("---------------------------------")
        print("Images type, Images length, 0th Image shape: ", type(imgs), "---", len(imgs),"---", imgs[0].shape)
        print("Labels Type, Labels length: ", type(labels), "---", len(labels), labels)



        if i == 7:
            images = imgs
            print(len(images), images.shape)
            for j in range(len(images)):
                z = images[j]
                l = labels[j]
                ax = plt.subplot(4, 4, (j + 1))
                plt.subplots_adjust(top=0.85)
                plt.title(str(l), fontsize=9)
                plt.axis('off')
                plt.imshow(z.permute(1, 2, 0))
        plt.savefig("/path/to/escape_data/classifier_data/classifier_data/example_dataset.png")





