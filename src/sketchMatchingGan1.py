from __future__ import print_function
import torch
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
netSketch.init_networks(weights_init)
netSketch.train()

print('--- training ---')
for epoch in range(opts.epochs):
    itr = 0
    fnames = load_train_batchfnames(opts.text_path, opts.batchsize, 
                                    opts.text_datasize, trainnum=opts.Btraining_num)
    fnames2 = load_train_batchfnames(opts.augment_text_path, opts.batchsize, 
                                    opts.augment_text_datasize, trainnum=opts.Btraining_num)
    for ii in range(len(fnames)):
        fnames[ii][0:int(opts.batchsize/2-1)] = fnames2[ii][0:int(opts.batchsize/2-1)]
    for fname in fnames:
        itr += 1
        t = prepare_text_batch(fname, anglejitter=True) #shape=[16,3,256,256]
        t = to_var(t) if opts.gpu else t
        losses = netSketch.one_pass(t, [l/4.-1. for l in range(0,9)])   #[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]  
        print('Epoch [%d/%d][%03d/%03d]' %(epoch+1, opts.epochs,itr,len(fnames)), end=': ')
        print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f'%(losses[0], losses[1], losses[2]))

print('--- save ---')
# directory
torch.save(netSketch.state_dict(), opts.save_GB_name)



netSketch.eval()
I = load_image('../data/style/leaf.png')
I = to_var(I[:,:,:,0:int(I.size(3)/2)])
result = netSketch(I, -1.)
visualize(to_data(result[0]))

