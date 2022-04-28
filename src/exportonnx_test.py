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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
import os

from pathlib import Path as ppath
FILE = ppath(__file__).resolve()
#ROOT = FILE.parents[1]
#if str(ROOT) not in sys.path:
#    sys.path.append(str(ROOT))
#ROOT = ppath(os.path.relpath(ROOT, ppath.cwd())) 
ROOT = FILE.parents[0]
cwdir = os.getcwd()
cudir = os.chdir(ROOT)

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
torch.onnx.export(model, (dummy_input,c), 'model.onnx', verbose=False)
