#
# https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model
#
from __future__ import print_function
import torch
from torch.autograd import Variable
from models import SketchModule
from utils import load_image, to_data, to_var, visualize, save_image, gaussian, weights_init
from utils import load_train_batchfnames, prepare_text_batch
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

from pathlib import Path as ppath
FILE = ppath(__file__).resolve()
#ROOT = FILE.parents[1]
#if str(ROOT) not in sys.path:
#    sys.path.append(str(ROOT))
#ROOT = ppath(os.path.relpath(ROOT, ppath.cwd())) 
ROOT = FILE.parents[0]
cwdir = os.getcwd()
cudir = os.chdir(ROOT)

opts = argparse.ArgumentParser()
opts.GB_nlayers = 6
opts.DB_nlayers = 5
opts.GB_nf = 32
opts.DB_nf = 32
opts.gpu = True
opts.epochs = 3
opts.save_GB_name = '../save/GB.ckpt'
opts.batchsize = 16
opts.text_path = '../data/rawtext/yaheiB/train'
opts.augment_text_path = '../data/rawtext/augment'
opts.text_datasize = 708
opts.augment_text_datasize = 5
opts.Btraining_num = 12800

# create model
print('--- create model ---')
netSketch = SketchModule(opts.GB_nlayers, opts.DB_nlayers, opts.GB_nf, opts.DB_nf, opts.gpu)
if opts.gpu:
    netSketch.cuda()
#netSketch.init_networks(weights_init)
#netSketch.train()


import torch.onnx 

#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    input_size = (16, 3, 256, 256)
    # dummy_input = Variable(torch.randn(input_size)).cuda()
    # dummy_input = torch.randn(1, input_size, requires_grad=True)   
    #dummy_input = torch.randn(32, 3, 256, 256, requires_grad=True).cuda() 
    #dummy_input = Variable(torch.randn(32, 3, 256, 256, requires_grad=True)).cuda()
    I = load_image('../data/style/leaf.png')
    I = to_var(I[:,:,:,0:int(I.size(3)/2)])
    #result = netSketch(I, -1.)

    # Export the model   
    torch.onnx.export(model,          # model being run 
         (I, -1), # dummy_input,       # model input (or a tuple for multiple inputs) 
         "ImageClassifier.onnx",      # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,    # whether to execute constant folding for optimization 
         input_names = ['modelInput'],     # the model's input names 
         output_names = ['modelOutput'],   # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},  # variable length axes
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

if __name__ == "__main__": 

    # Let's build our model 
    #train(5) 
    #print('Finished Training') 

    # Test which classes performed well 
    #testAccuracy() 

    # Let's load the model we just created and test the accuracy per label 
    #netSketch.eval()
    #I = load_image('../data/style/leaf.png')
    #I = to_var(I[:,:,:,0:int(I.size(3)/2)])
    #result = netSketch(I, -1.)

    model = netSketch
    #path = "myFirstModel.pth" 
    #model.load_state_dict(torch.load(path)) 
    state_dict = torch.load('../save/GB.ckpt')
    model.load_state_dict(state_dict)

    

        
    # Test with batch of images 
    #testBatch() 
    # Test how the classes performed 
    #testClassess() 
 
    # Conversion to ONNX 
    Convert_ONNX() 


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
import os

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):

    def __init__(self, conv_dim=64, c_dim=10, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2,
                                    kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim //
                                             2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


model = Generator().cuda()
state_dict = torch.load('../save/GB.ckpt')
model.load_state_dict(state_dict, strict=False)
dummy_input = Variable(torch.randn(32, 3, 256, 256)).cuda()


dummy_input = Variable(torch.randn(32, 3, 256, 256)).cuda()
torch.onnx.export(model, dummy_input, 'model.onnx', verbose=False)
"""