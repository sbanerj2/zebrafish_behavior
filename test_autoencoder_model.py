'''
Code for testing reconstruction on the trajectory test dataset
Dataset: https://drive.google.com/drive/folders/1mwZJ06zerm9-z7Q606xPip1SiTnxduqY?usp=sharing
The test images are kept in the file 'test_list.txt'
Please change the paths
to run: python test_autoencoder_model.py
Needs GPU
'''


import os
import torch
import numpy as np
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as TF
from torchvision.transforms import ToPILImage, Compose, ToTensor, CenterCrop
from datasets_autoencoder import TrajDataset, transform
#import eval_seg as es
#from transform import Scale
from PIL import Image
from changed_model import *
from torchsummary import summary
import matplotlib.pyplot as plt



# CUDA for PyTorch
base_directory = '/path/to/datasets/escape_data/autoencoder_data'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


## Parameters
batch_size = 1

## Load Test Dataset
img_file = 'test_list.txt'
img_list = [line.rstrip('\n') for line in open(os.path.join(base_directory, img_file))]
print("--------------------------------------------------------")
print ("length of test dataset: ", len(img_list)) ##Debug statements
print("--------------------------------------------------------")

test_set = TrajDataset(img_list, transform, split='test')
print("--------------------------------------------------------")
print("Length of test data: ", len(test_set))
test_loader = data.DataLoader(test_set, batch_size=batch_size,
                             num_workers=8)

## Model

## Initialize model
model = SegModel(training=False)
#if torch.cuda.device_count() > 1:
#    model = nn.DataParallel(model)
model.to(device)
print(model)

#encoder = VGG19_Encoder()
#encoder.to(device)
#summary(model, (3, 256, 256))

## Load saved model weights
load_path = '/path/to/model_weights/model_best128_0603.pth.tar'
checkpoint = torch.load(load_path)
epoch = checkpoint['epoch']
train_step = checkpoint['train step']
train_loss = checkpoint['train_loss']
val_loss = checkpoint['val_loss']
print ("------Epoch {} | Step {} | Train Loss {} | Validation Loss {} -----------".format(epoch, train_step, train_loss, val_loss))
model.load_state_dict(checkpoint['state_dict'])
print ("model loaded")

model.eval()
encoder = model.features
encoder.to(device)

out_dir = '/path/to/save/results/autoencoder'

##For the test dataset
for step, data in enumerate(test_loader):
    images, _ = data
    images = images.to(device)
    enc_feat = encoder(images)
    print("encoded feature shape in test: ", enc_feat.shape, type(enc_feat))
    print("encoded feature :", enc_feat)
    output_4 = model(images)
    output_4 = np.squeeze(output_4)
    output = output_4.cpu()
    output = output.permute(1,2,0)
    output = output.detach().numpy()
    print("output details:", type(output), output.shape, np.unique(output))
    new_image = output*255
    new_image = new_image.astype(np.uint8)
    #print("new_image details: ", type(new_image), new_image.shape, np.unique(new_image))
    #print("new_image now: ", type(new_image), new_image.shape, np.unique(new_image))
    img = Image.fromarray(new_image, 'RGB')
    img_name = "test_64_" + str(step) + ".png"
    if not os.path.exists(os.path.join(out_dir, 'result')):
        os.makedirs(os.path.join(out_dir, 'result'))
    img.save( os.path.join(out_dir, 'result', img_name))
    #print ("%s saved" % str(img_name))
    ## save original
    org = images *255
    org = org.long()
    org = org.cpu().numpy()
    org = org.astype(np.uint8)
    #print ("Original image shape: ", org.shape)
    org = np.squeeze(org)
    #print ("Original image shape now: ", org.shape)
    org = np.transpose(org, (1, 2, 0))
    #print ("Original image shape after: ", org.shape)
    org = Image.fromarray(org, 'RGB')
    img_orig = "orig_" + str(step) + ".png"









