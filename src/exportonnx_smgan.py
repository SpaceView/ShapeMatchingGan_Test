from __future__ import print_function
import torch
from models import SketchModule, ShapeMatchingGAN
from utils import load_image, to_data, to_var, visualize, save_image, gaussian, weights_init
from utils import load_train_batchfnames, prepare_text_batch, load_style_image_pair, cropping_training_batches
import random
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
# SMGAN
opts.GS_nlayers = 6
opts.DS_nlayers = 4
opts.GS_nf = 32
opts.DS_nf = 32
opts.GT_nlayers = 6
opts.DT_nlayers = 4
opts.GT_nf = 32
opts.DT_nf = 32

# SketchModule
opts.GB_nlayers = 6
opts.DB_nlayers = 5
opts.GB_nf = 32
opts.DB_nf = 32
opts.load_GB_name = '../save/GB-iccv.ckpt'

# train 
opts.gpu = True
opts.step1_epochs = 30
opts.step2_epochs = 40
opts.step3_epochs = 80
opts.step4_epochs = 10
opts.batchsize = 16
opts.Straining_num = 2560
opts.scale_num = 4
opts.Sanglejitter = True
opts.subimg_size = 256
opts.glyph_preserve = False
opts.text_datasize = 708
opts.text_path = '../data/rawtext/yaheiB/train'

# data and path
opts.save_path = '../save/'
opts.save_name = 'maple'
opts.style_name = '../data/style/maple.png'


# create model
print('--- create model ---')
netShapeM = ShapeMatchingGAN(opts.GS_nlayers, opts.DS_nlayers, opts.GS_nf, opts.DS_nf,
                 opts.GT_nlayers, opts.DT_nlayers, opts.GT_nf, opts.DT_nf, opts.gpu)

if opts.gpu:
    netShapeM.cuda()
#netShapeM.init_networks(weights_init)
#netShapeM.train()

import torch.onnx 
from torch.autograd import Variable

#Function to Convert to ONNX 
def Convert_ONNX(model, model_name, dummy_input): 

    # set the model to inference mode 
    model.eval()

    # Let's create a dummy input tensor  
    #dummy_input = torch.randn(32, 3, 256, 256, requires_grad=True).cuda() 
    #dummy_input = Variable(torch.randn(32, 3, 256, 256, requires_grad=True)).cuda()
    #I = load_image('../data/style/leaf.png')
    #I = to_var(I[:,:,:,0:int(I.size(3)/2)])
    #result = netSketch(I, -1.)

    # Export the model   
    torch.onnx.export(model,          # model being run 
        dummy_input, # (I, -1),       # model input (or a tuple for multiple inputs) 
        model_name,  # "GB-ckpt1.onnx",      # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=10,    # the ONNX version to export the model to 
        do_constant_folding=True,    # whether to execute constant folding for optimization 
        input_names = ['modelInput'],     # the model's input names 
        output_names = ['modelOutput'],   # the model's output names 
        dynamic_axes={'modelInput' : {0 : 'batch_size'},  # variable length axes
                               'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX: ', model_name) 

if __name__ == "__main__": 
    """
    # done ---- OK ---- 20220428
    model = netShapeM.G_S
    state_dict = torch.load('../save/maple-GS-iccv.ckpt')
    model.load_state_dict(state_dict) 
    I = load_image('../data/rawtext/yaheiB/val/0801.png')
    I = to_var(I[:,:,32:288,32:288])
    I[:,0:1] = gaussian(I[:,0:1], stddev=0.2)
    dummy_input = (I, 1.0)
    Convert_ONNX(model, 'maple-GS-iccv_ckpt.onnx', dummy_input)
    """

    
    state_dict = torch.load('../save/maple-GT-iccv.ckpt')
    model = netShapeM.G_T
    model.load_state_dict(state_dict)
    I = Variable(torch.randn(1, 3, 320, 320, requires_grad=True)).cuda()
    Convert_ONNX(model, 'maple-GT-iccv_ckpt.onnx', I)

print("all done!")