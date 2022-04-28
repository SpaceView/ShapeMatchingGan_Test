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
netSketch = SketchModule(opts.GB_nlayers, opts.DB_nlayers, opts.GB_nf, opts.DB_nf, opts.gpu)

if opts.gpu:
    netShapeM.cuda()
    netSketch.cuda()
netShapeM.init_networks(weights_init)
netShapeM.train()

netSketch.load_state_dict(torch.load(opts.load_GB_name))
netSketch.eval()

print('--- training ---')
# load image pair
scales = [l*2.0/(opts.scale_num-1)-1 for l in range(opts.scale_num)]
Xl, X, _, Noise = load_style_image_pair(opts.style_name, scales, netSketch, opts.gpu)
Xl = [to_var(a) for a in Xl] if opts.gpu else Xl
X = to_var(X) if opts.gpu else X
Noise = to_var(Noise) if opts.gpu else Noise
for epoch in range(opts.step1_epochs):
    for i in range(int(opts.Straining_num/opts.batchsize)):
        idx = opts.scale_num-1
        xl, x = cropping_training_batches(Xl[idx], X, Noise, opts.batchsize, 
                                  opts.Sanglejitter, opts.subimg_size, opts.subimg_size)
        losses = netShapeM.structure_one_pass(x, xl, scales[idx])
        print('Step1, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.step1_epochs, i+1, 
                                                      opts.Straining_num/opts.batchsize), end=': ')
        print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lgly: %+.3f'%(losses[0], losses[1], losses[2], losses[3]))
netShapeM.G_S.myCopy()
for epoch in range(opts.step2_epochs):
    for i in range(opts.Straining_num/opts.batchsize):
        idx = random.choice([0, opts.scale_num-1])
        xl, x = cropping_training_batches(Xl[idx], X, Noise, opts.batchsize, 
                                  opts.Sanglejitter, opts.subimg_size, opts.subimg_size)
        losses = netShapeM.structure_one_pass(x, xl, scales[idx])
        print('Step2, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.step2_epochs, i+1, 
                                                      opts.Straining_num/opts.batchsize), end=': ')
        print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lgly: %+.3f'%(losses[0], losses[1], losses[2], losses[3]))
for epoch in range(opts.step3_epochs):
    for i in range(opts.Straining_num/opts.batchsize):
        idx = random.choice(range(opts.scale_num))
        xl, x = cropping_training_batches(Xl[idx], X, Noise, opts.batchsize, 
                                  opts.Sanglejitter, opts.subimg_size, opts.subimg_size)
        losses = netShapeM.structure_one_pass(x, xl, scales[idx])  
        print('Step3, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.step3_epochs, i+1, 
                                                      opts.Straining_num/opts.batchsize), end=': ')
        print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lgly: %+.3f'%(losses[0], losses[1], losses[2], losses[3]))
if opts.glyph_preserve:
    fnames = load_train_batchfnames(opts.text_path, opts.batchsize, 
                                    opts.text_datasize, opts.Straining_num)
    for epoch in range(opts.step4_epochs):
        itr = 0
        for fname in fnames:
            itr += 1
            t = prepare_text_batch(fname, anglejitter=False)
            idx = random.choice(range(opts.scale_num))
            xl, x = cropping_training_batches(Xl[idx], X, Noise, opts.batchsize, 
                                      opts.Sanglejitter, opts.subimg_size, opts.subimg_size)
            t = to_var(x) if opts.gpu else t
            losses = netShapeM.structure_one_pass(x, xl, scales[idx], t)  
            print('Step4, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.step4_epochs, itr+1, 
                                                      len(fnames)), end=': ')
            print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lgly: %+.3f'%(losses[0], losses[1], losses[2], losses[3])) 
        
print('--- save ---')
# directory
netShapeM.save_structure_model(opts.save_path, opts.save_name)



netShapeM.eval()
I = load_image('../data/rawtext/yaheiB/val/0801.png')
I = to_var(I[:,:,32:288,32:288])
I[:,0:1] = gaussian(I[:,0:1], stddev=0.2)
result = netShapeM.G_S(I, 1.0)
visualize(to_data(result[0]))

