import torch
import torch.nn as nn
import torchvision
from collections import namedtuple
from torchsummary import summary
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math

## Autoencoder model definition file
## This is a model without skip connections as we are not interested 
## in exact replication or preserving final details of the input. 


'''Changing nn.Sequential to have multiple inputs'''
class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


def conv_operation(in_ch,out_ch):
    convolution = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding=1, kernel_size=3, stride=1),
        nn.ReLU(),
    )
    return convolution

'''Most of the print statements are for debugging purpose'''
class VGG19_Encoder(nn.Module):
    def __init__(self):
        super(VGG19_Encoder, self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)
        self.Conv1_features = list(vgg_model.features)[:4]
        self.Conv2_features = list(vgg_model.features)[5:9]
        self.Conv3_features = list(vgg_model.features)[10:18]
        self.Conv4_features = list(vgg_model.features)[19:27]
        self.Conv5_features = list(vgg_model.features)[28:35]
        self.Conv1 = nn.Sequential(*self.Conv1_features)
        self.Conv2 = nn.Sequential(*self.Conv2_features)
        self.Conv3 = nn.Sequential(*self.Conv3_features)
        self.Conv4 = nn.Sequential(*self.Conv4_features)
        self.Conv5 = nn.Sequential(*self.Conv5_features)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.MaxPool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.Linear1  = nn.Linear(in_features=32768, out_features=4096, bias=True)
        self.Linear_e2  = nn.Linear(in_features=4096, out_features=256, bias=True)
        self.Linear_e3  = nn.Linear(in_features=256, out_features=128, bias=True)
        self.Linear_e4  = nn.Linear(in_features=128, out_features=64, bias=True)


    def forward(self, x):
        N, channels, height, width = x.size()
        out1     = self.Conv1(x)
        out1_m   = self.MaxPool1(out1)
        out2     = self.Conv2(out1_m)
        out2_m   = self.MaxPool2(out2)
        out3     = self.Conv3(out2_m)
        out3_m   = self.MaxPool3(out3)
        out4     = self.Conv4(out3_m)
        out4_m   = self.MaxPool4(out4)
        out5  = self.Conv5(out4_m)
        out5_m = self.MaxPool5(out5)
        out5_m = out5_m.view(-1, 512 * 8 * 8)
        encoded = self.Linear1(out5_m)
        encoded = self.Linear_e2(encoded)
        encoded = self.Linear_e3(encoded)
        encoded = self.Linear_e4(encoded)

        return encoded



class Decoder(nn.Module):
    def __init__(self, init_weight):
        super(Decoder, self).__init__()


        ## To be used for calculating multi-scale loss
        self.conv2_1_4 = nn.Conv2d(512, 3, padding=0, kernel_size=1, stride=1)
        self.conv2_2_3 = nn.Conv2d(256, 3, padding=0, kernel_size=1, stride=1)
        self.conv2_3_2 = nn.Conv2d(128, 3 , padding=0, kernel_size=1, stride=1)
        self.conv2_4_1 = nn.Conv2d(64, 3, padding=0, kernel_size=1, stride=1)

        ## 3x3 convolution with ReLU()
        self.conv512 = conv_operation(512, 512)
        self.conv256 = conv_operation(256, 256)
        self.conv128 = conv_operation(128, 128)
        self.conv64  = conv_operation(64, 64)
        
        ## Linear layer
        self.linear_d3 = nn.Linear(in_features=64, out_features=128, bias=True)
        self.linear_d2 = nn.Linear(in_features=128, out_features=256, bias=True)
        self.linear_d1 = nn.Linear(in_features=256, out_features=4096, bias=True)
        self.linear2 = nn.Linear(in_features=4096, out_features=32768, bias=True)


        ## Deconvolution Layers:

        self.deconv1_512 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.deconv2_256 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3,stride=2, padding=1)
        self.deconv3_128 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.deconv4_64 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.Relu = nn.ReLU()

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        height = 256
        width = 256

        H16 = math.ceil(height / 16)
        W16 = math.ceil(width/ 16)
        H8 = 2* H16
        W8 = 2* W16
        H4 = 2* H8
        W4 = 2* W8
        H2 = 2* H4
        W2 = 2* W4
        H  = 2* H2
        W  = 2* W2


        x = self.linear_d3(x)
        x = self.linear_d2(x)
        x = self.linear_d1(x)
        x = self.linear2(x)
        x = x.view(-1, 512,8,8)

        d_out_1_1 = self.deconv1_512(x, output_size=(H16, W16))
        d_out_1 = self.deconv1_512(d_out_1_1, output_size=(H8, W8))
        decode_out1 = self.Relu(d_out_1)
        decode_out1 = self.conv512(decode_out1)
        decode_out1 = self.conv512(decode_out1)
        decode_out1 = self.conv512(decode_out1)
        d_out_2 = self.deconv2_256(decode_out1, output_size=(H4, W4))
        decode_out2 = self.Relu(d_out_2)
        decode_out2 = self.conv256(decode_out2)
        decode_out2 = self.conv256(decode_out2)
        decode_out2 = self.conv256(decode_out2)
        d_out_3 = self.deconv3_128(decode_out2, output_size=(H2, W2))
        decode_out3 = self.Relu(d_out_3)
        decode_out3 = self.conv128(decode_out3)
        decode_out3 = self.conv128(decode_out3)
        d_out_4 = self.deconv4_64(decode_out3, output_size=(H, W))
        decode_out4 = self.Relu(d_out_4)
        decode_out4 = self.conv64(decode_out4)
        decode_out4 = self.conv64(decode_out4)
        decode_out4_loss = self.conv2_4_1(decode_out4)

        return decode_out4_loss


class SegModel(nn.Module):
    def __init__(self, training):
        super().__init__()
        self.training   = training
        print ("In SegModel: ", self.training)

        if self.training:
            self.init_weight = True
        else:
            self.init_weight = False

        self.features = VGG19_Encoder()
        self.decoder = Decoder(self.init_weight)


    def forward(self, x):
        x= self.features(x)
        x= self.decoder(x)
        return x

        



if __name__ == '__main__': #for debugging
    print("---------------------------------")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("---------------------------------")
    print ("SegModel ")
    model2 = SegModel(training=False)
    model2.to(device)
    summary(model2, (3, 256, 256))
    print("Whole Model loaded")
    print("---------------------------------")
    print("Only Features")
    model3 = model2.features
    model3.to(device)
    print(model3)
    summary(model3, (3, 256, 256))
    
    
















