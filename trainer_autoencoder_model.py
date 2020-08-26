'''
Code for testing reconstruction on the trajectory test dataset
Dataset can be found at the following link:
https://drive.google.com/drive/folders/1mwZJ06zerm9-z7Q606xPip1SiTnxduqY?usp=sharing
The training images are kept in the file 'train_list.txt'
Please change the paths
to run: python trainer_autoencoder_model.py
'''



import os
import sys
import torch
import shutil
import numpy as np
from random import sample
from torch.utils import data
import torch.nn as nn
from sklearn.model_selection import train_test_split
from datasets_autoencoder import TrajDataset, transform
from torch.utils.data import Dataset, DataLoader
from changed_model import *
from torch import optim
import matplotlib.pyplot as plt




def save_checkpoint(state, is_best,
                    filename='/path/to/model_weights/checkpoint_0603.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        '/path/to/model_weights/model_best128_0603.pth.tar')
        print ("Best model saved")


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


# CUDA for PyTorch
base_directory = '/path/to/datasets/escape_data/autoencoder_data'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Parameters
train_params = {'batch_size': 10,
                'shuffle': True,
                'num_workers': 12}
val_params = {'batch_size': 10,
              'shuffle': False,
              'num_workers': 12}

max_epochs = 100
lr_decoder = 0.001
lr_encoder = 0.00001

## Initialize model
model = SegModel(training=True)
model.to(device)

## No. of parameters in the model
params = list(model.parameters())

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_encoder)

# Datasets
## Dividing the entire Training set as a train set (90%)&a validation set(10%)

img_file = 'train_list.txt'
img_list = [line.rstrip('\n') for line in open(os.path.join(base_directory, img_file))]
print("--------------------------------------------------------")
print ("length of trainval dataset: ", len(img_list))  ##Debug statements
print("--------------------------------------------------------")
img_shuffled = sample(img_list, len(img_list))
train_ind = int(0.9 * len(img_shuffled))
train_lst = img_shuffled[:train_ind]
print("--------------------------------------------------------")
print("First 10 training images", train_lst[:10])  # Debug statements
print("--------------------------------------------------------")
val_lst = img_shuffled[train_ind:]

# Dataset initialization & creating dataloader
training_set = TrajDataset(train_lst, transform, split='train')
print("--------------------------------------------------------")
print("Length of training data: ", len(training_set))
train_loader = DataLoader(training_set, **train_params)
print("No. of batches for training", len(train_loader))
print("--------------------------------------------------------")
validation_set = TrajDataset(val_lst, transform, split='val')
print("Length of validation data: ", len(validation_set))
val_loader = DataLoader(validation_set, **val_params)
print("No. of batches for validation", len(val_loader))
print("--------------------------------------------------------")


best_loss = 10000
# Loop over epochs
for epoch in range(0, max_epochs):
    # Training
    for step, data in enumerate(train_loader):
        images, _ = data
        images = images.to(device)
        print("Epoch {} Iteration {} :".format(epoch, step))
        total_iterations = (step + 1) * (epoch + 1)
        # print ("Total Iterations so far:", total_iterations)
        # print("--------------------------------------------------------")
        # print("local_batch image shape", local_batch.shape, len(local_batch), type(local_batch))
        # print("local_batch label shape", local_labels.shape, len(local_labels), type(local_labels))
        # print("--------------------------------------------------------")

        out_images = model(images)
        # print("--------------------------------------------------------")
        print("decoder output:", out_images.shape)
        # print("local_labels output: ", local_labels.shape)
        # print("--------------------------------------------------------")
        loss = loss_function(out_images, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 150 == 0:  ##1492 images with a batch size of 10, no. of batches per epoch is 149+ 1(with 2 images)
            print("--------------------------------------------------------")
            print("{} epoch completed: ".format((epoch + 1)))
            print("Training loss is: ", loss)
            print("--------------------------------------------------------")
            val_loss_all = []
            # Validation
            for val_step, (val_batch, val_labels) in enumerate(val_loader):
                # Transfer to GPU
                val_batch = val_batch.to(device)

                # print("Validation step: ", val_step)
                # print("val_labels output shape: ", val_labels.shape, type(val_labels))
                val_4 = model(val_batch)
                val_loss = loss_function(val_4, val_batch)
                val_loss_all.append(val_loss.data.cpu().numpy())

            val_loss_actual = sum(val_loss_all) / len(val_loss_all)


            print('Epoch: ', (epoch + 1), '| train loss: %.4f' % loss.data.cpu().numpy(),
                  '| val loss: %.4f' % val_loss_actual)
            print("--------------------------------------------------------")
            # torch.save(model, './model/model_e{}_s{}.pkl'.format(epoch, step))
            print("Best validation loss before: ", best_loss)
            is_best = val_loss_actual < best_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'train step': step,
                'train_loss': loss.data.cpu().numpy(),
                'val_loss': val_loss_actual,
                # 'val_loss' : val_loss.data.cpu().numpy(),
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best)
            print("--------------------------------------------------------")
            if is_best:
                best_loss = val_loss_actual
                print("Best validation loss of {} at epoch{}. ".format(best_loss, epoch))
