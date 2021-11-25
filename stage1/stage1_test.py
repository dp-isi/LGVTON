import sys 
sys.path.append('./')
import params as p
import os
os.environ['CUDA_VISIBLE_DEVICES']=str(p.gpu_id)
import tensorflow as tf
import models.my_layers as my_layers
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import load_model
import stage1.tps_transform as tps
import matplotlib.pyplot as plt
from models import stage1_model
import matplotlib.image as mpimg
import numpy as np
import shutil
from PIL import Image
from data import dataloader_test as generate
import keras.backend as K
from datetime import datetime
import argparse
# ------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-batch_size',default='1',help='size of batch to predict',type=int )
parser.add_argument('-train_test',default='test',help='execute on train or test dataset' )
parser.add_argument('-dataset',default='df',help='dataset' )
args = parser.parse_args()
bs  = args.batch_size
data_type = args.train_test
dataset = args.dataset

# --------------------------------------

model, model_final = stage1_model.get_model_v3(compile_model=True)

pathm = './checkpoints/stage1/'

def model_load():
	model_number='latest'

	custom_objects = {"FeatureCorrelation": my_layers.FeatureCorrelation(out_dim=36), "InstanceNormalization": InstanceNormalization}

	model, _ = stage1_model.get_model_v3()
	model = load_model('%smodel_%s.h5'%(pathm,model_number), custom_objects)
	return model

model = model_load()



path='./test_train_pairs/'
if(data_type=='test'):
	filename = 'test_pairs.txt'
else:
	filename = 'train_pairs.txt'

f = open(path+filename,'r')
dc = len(f.readlines());f.close()

path_res = p.path_stage1_test 


if(os.path.exists(path_res)):
	shutil.rmtree(path_res)
	os.makedirs(path_res)
else:
	os.makedirs(path_res)


path_warps=path_res+'warps/'
os.mkdir(path_warps)
path_dis=path_res+'display/'
os.mkdir(path_dis)


def warp(l_src_flm,l_trgt_flm_pred,l_src_hlm,l_trgt_hlm,l_src_cl,l_src_cl_m):

	def merge(l1,l2):
		l1= l1.reshape(-1,2)
		l2 = l2.reshape(-1,2)
		l=np.zeros(shape=(l1.shape[0]+l2.shape[0],2),dtype='float32')
		l[:l1.shape[0],:]=l1
		l[l1.shape[0]:,:]=l2
		return l

	l_warp_cloth=[]
	l_warp_cloth_m=[]

	for i in range(len(l_trgt_flm_pred)):
		
		l = merge(l_src_hlm[i],l_src_flm[i])
		l1 = merge(l_trgt_hlm[i],l_trgt_flm_pred[i])
		
		temp = tps.warp_image_cv(l_src_cl[i], l, l1, dshape=(256, 256))
		temp = np.clip(temp,0,1)

		temp_m = tps.warp_image_cv(l_src_cl_m[i], l, l1, dshape=(256, 256))		
		temp_m = (temp_m>0.9).astype('float32')
		temp_m = np.clip(temp_m,0,1)

		l_warp_cloth.append(temp)
		l_warp_cloth_m.append(temp_m)

	return (l_warp_cloth,l_warp_cloth_m)




obj_test = generate.generate_data(batch_size=bs,stage=1,dataset=dataset)





exec_time=[]
for t in range(0,dc,bs):
	print(t,bs)
	
	[l_src_hlm,l_trgt_hlm,l_src_flm],[l_warp_cloth,l_warp_cloth_mask],[l_src_img,l_trgt_img,l_name] = next(obj_test)

	l_trgt_flm_pred = model.predict([l_src_hlm,l_trgt_hlm,l_src_flm])

	l_warp,l_warp_mask = warp(l_src_flm,l_trgt_flm_pred,l_src_hlm,l_trgt_hlm,l_warp_cloth,l_warp_cloth_mask)

	for j in range(bs):
		
		name = l_name[j]
		
		m=np.clip(l_warp_mask[j],0,1)
		
		c = np.clip(l_warp[j],0,1)
		
		c_wbkg = np.ones(shape=c.shape)*(1-m) + c*m

		Image.fromarray((c_wbkg*255).astype('uint8')).save('%s/%s'%(path_warps,name.replace('.jpg','_cloth_wbkg.jpg')))
		Image.fromarray((c*255).astype('uint8')).save('%s/%s'%(path_warps,name.replace('.jpg','_cloth_bbkg.jpg')))
		Image.fromarray((m*255).astype('uint8')).save('%s/%s'%(path_warps,name.replace('.jpg','_cloth_mask.png')))


		flm1 = (l_src_flm[j].reshape(-1,2)*255).astype('uint8')
		flm2 = (l_trgt_flm_pred[j].reshape(-1,2)*255).astype('uint8')
		
		plt.subplot(1,3,1)
		plt.imshow(l_src_img[j])
		plt.scatter(flm1[:,0],flm1[:,1],c=['red','green','blue','purple','orange','yellow'])
		
		plt.subplot(1,3,2)
		plt.imshow(l_trgt_img[j])
		plt.scatter(flm2[:,0],flm2[:,1],c=['red','green','blue','purple','orange','yellow'])
		
		plt.subplot(1,3,3)
		plt.imshow(l_warp[j])
		save_name= '%s/%s'%(path_dis,name.replace('.jpg','_display.jpg'))
		plt.savefig(save_name)
		plt.close()

