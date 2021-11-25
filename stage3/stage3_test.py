import sys 
sys.path.append('./')
import params as p
import os
os.environ['CUDA_VISIBLE_DEVICES']=str(p.gpu_id)
import tensorflow as tf	
import matplotlib.image as mpimg
import numpy as np
import shutil
from PIL import Image
from data import dataloader_test as generate_test
import keras.backend as K
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import load_model
from models import stage3_model
import argparse
# ------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-batch_size',default='1',help='size of batch to predict',type=int )
parser.add_argument('-train_test',default='test',help='execute on train or test dataset' )
parser.add_argument('-dataset',default='mpv',help='dataset' )
args = parser.parse_args()
bs  = args.batch_size
data_type = args.train_test
dataset = args.dataset

stage=3
pathm = './checkpoints/stage%d/'%(stage)
paths = p.path_stage3_test


if(data_type=='test'):
	paths=paths+'test/'
if(not os.path.exists(paths)):
	os.makedirs(paths)


path='./test_train_pairs/'
data_type='test'
if(data_type=='test'):
	filename = 'test_pairs.txt'
else:
	filename = 'train_pairs.txt'

f = open(path+filename,'r')
dc = len(f.readlines());f.close()


generator,gan, discriminator = stage3_model.build_gan()

gan.load_weights('%sgan_wt_latest.h5'%(pathm))

path_res = './results/stage%d/'%(stage)

obj_test_test = generate_test.generate_data(batch_size=bs,stage=stage,dataset=dataset,womask=0)


def f1(a):

	a = (a+1)/2.0
	m = (np.clip(a,0,1)*255).astype('uint8')

	return m

for t in range(0,dc,bs):
	print(t)
	
	[l_im_masked,l_warp_cloth,l_iuv,l_z,l_prev_cloth_mask],[l_name] =next(obj_test_test)
	input_list = [l_im_masked,l_iuv,l_prev_cloth_mask,l_warp_cloth,l_z]
	
	a,b = gan.predict(input_list)

	for j in range(bs):
			
		name = l_name[j]
		m = f1(a[j])
				
		Image.fromarray(m).save(paths+name)

