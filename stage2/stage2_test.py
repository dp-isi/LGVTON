import sys 
sys.path.append('./')
import params as p
import os
os.environ['CUDA_VISIBLE_DEVICES']=str(p.gpu_id)
import tensorflow as tf
from PIL import Image
from data import dataloader_test as generate
import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import load_model
import matplotlib.image as mpimg
import numpy as np
import shutil
from models import stage2_model
import argparse
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-batch_size',default='1',help='size of batch to predict',type=int )
parser.add_argument('-train_test',default='test',help='execute on train or test dataset' )
parser.add_argument('-dataset',default='df',help='dataset' )
args = parser.parse_args()
bs  = args.batch_size
data_type = args.train_test
dataset = args.dataset


stage=2
pathm = './checkpoints/stage%d/'%(stage)

def model_load():
	model = stage2_model.build_gan()
	model.load_weights('%smodel_wt_%s.h5'%(pathm,'latest'))
	return model

model = model_load()


path='./test_train_pairs/'
if(data_type=='test'):
	filename = 'test_pairs.txt'
else:
	filename = 'train_pairs.txt'

f = open(path+filename,'r')
dc = len(f.readlines());f.close()


paths = p.path_stage2_test

if(not os.path.exists(paths)):
	os.makedirs(paths)

exec_time=[]


obj_test = generate.generate_data(batch_size=bs,stage=stage,dataset=dataset)

for t in range(0,dc,bs):

	[l_warp_cloth,l_iuv],[l_name] = next(obj_test)

	a = model.predict([l_warp_cloth,l_iuv])

	for j in range(bs):
		
		name = l_name[j]
		
		m = (np.clip(a[j],0,1)*255).astype('uint8')
		
		Image.fromarray(m).save(paths+name)


