
import params as p
import numpy as np
from PIL import Image
import random
import pickle
import os 
import cv2
import matplotlib.image as mpimg
import json


f = open('%sdict_fashion_landmarks.pickle'%(p.data_path),'rb')
lm = pickle._Unpickler(f)
lm.encoding = 'latin1'
lm=lm.load()
f.close()


def get_hlm(line,dataset):
	if(dataset=='mpv'):
		line_hlm = line.replace('all','all_person_clothes_keypoints').replace('.jpg','_keypoints.json')
	else:
		line_hlm = line.replace('Img/img','pose/pose_json').replace('.jpg','.json')
	
	obj = json.load(open(line_hlm))
	data=np.array(obj['people'][0]['pose_keypoints'])
	data=np.reshape(data,(18,3))
	keypoint_indices = [1,2,3,4,5,6,7,8,11]
	keypoints = np.copy(data[keypoint_indices][:,:2])
	if(dataset=='mpv'):
		keypoints[:,0] = keypoints[:,0]+32
	return keypoints

def rng11(data):#range tanh
	data = (data*2)-1
	return data


def generate_data(batch_size = 5, shuffle=False, filename='test_pairs.txt',dataset='df',stage=1,womask=0,woflm=0):


	f = open('test_train_pairs/%s'%(filename))
	lines=f.readlines()
	f.close()

	index=-1

	while(1):		

		l_src_hlm,l_trgt_hlm,l_src_flm,l_trgt_flm=[],[],[],[]
		l_src_img,l_trgt_img=[],[]

		l_warp_cloth,l_warp_cloth_mask,l_iuv,l_prev_cloth_mask=[],[],[],[]

		l_im_masked,l_im_masked_wocl,l_bbox_mask = [],[],[]

		l_z=[]

		l_name=[]

		for i in range(batch_size):
		
			index=(index+1)%len(lines)
			if(dataset=='mpv'):
				base_name1,base_name2= lines[index].split(' ')
			else:
				base_name1,base_name2 = lines[index].split(' ')
			base_name2 = base_name2[:-1]

			src_line,trgt_line = base_name1,base_name2

			if(dataset =='mpv'):
				# print('dataset-----------------------mpv')
				src_line=p.data_path + 'all/' +src_line.replace('+','/')
				trgt_line=p.data_path + 'all/' + trgt_line.replace('+','/')
			else:#df				
				src_line=p.data_path +'Img/img/'+ src_line
				trgt_line=p.data_path + 'Img/img/' + trgt_line	
				# print('dataset-----------------------df',src_line,trgt_line)		
			
			def resize_img_forMPV(im,bkg=''):
				type=1
				
				if(len(im.shape)>2):
					dim=(256,256,im.shape[-1])
					
				else:
					dim=(256,256)
					type=0
				
				mult,dtyp=(255,'uint8') if(im.max()>1) else (1,'float32')
				temp = np.ones(shape=dim,dtype=dtyp)*mult
				if(bkg=='black'):
					temp=temp*0
				# print(im.shape)
				if(type==1):
					temp[:,32:-32,:] = im
				else:
					temp[:,32:-32] = im				
				return temp

			if(dataset=='mpv'):
				src_img=resize_img_forMPV(np.array(Image.open(src_line)))
				trgt_img=resize_img_forMPV(np.array(Image.open(trgt_line)))
			else:
				src_img=np.array(Image.open(src_line))
				trgt_img=np.array(Image.open(trgt_line))

			
			# print(src_line,trgt_line)

			result_name = base_name1.replace('/','=').replace('.jpg','')+ '_TO_' + base_name2.replace('/','=')
			# result_name = base_name2.replace('/','=').replace('.jpg','')+ '_TO_' + base_name2.replace('/','=')
			l_name.append(result_name)

			def f_iuv():
				if(dataset=='df'):
					iuv_name=(trgt_line.replace('Img/img','Densepose')).replace('.jpg','_IUV.png')
					# print(iuv_name)
				else:
					iuv_name=trgt_line.replace('.jpg','_IUV.png')
				if(not os.path.exists(iuv_name)):			   
					return _,-1 
				if(dataset=='mpv'):
					iuv = resize_img_forMPV(cv2.imread(iuv_name),bkg='black')
				else:
					iuv = cv2.imread(iuv_name)
				iuv = iuv.astype('float32')

				iuv[:,:,1:] = iuv[:,:,1:]/255.0
				iuv[:,:,:1] = iuv[:,:,:1]/24.0
				# print(iuv_name)
				return iuv,1

			def f_warp_cloth(stage):
				if(stage==1):
					fname=src_line
					sname=src_line
					if(dataset=='mpv'):
						seg = resize_img_forMPV(np.array(Image.open(sname.replace('all','all_parsing').replace('.jpg','.png'))))
					else:
						seg = np.array(Image.open(fname.replace('Img/img','cloth_parsing').replace('.jpg','.png')))
						
					
					cl_mask = (seg==5) + (seg==7) + (seg==6)
				
					cl_mask_3 = np.tile((((seg==5) + (seg==7) + (seg==6))[:,:,np.newaxis]),(1,1,3)).astype('float32') #chk 6 
					
					warp_cloth = (src_img/255.0)*(cl_mask_3)
					
					warp_cloth_mask = cl_mask_3
					
					return warp_cloth,warp_cloth_mask

				elif(stage==2 or stage==3):
					temp=np.array(Image.open(p.path_stage1_test+'warps/'+result_name.replace('.jpg','_cloth_mask.png')))				
					warp_cloth_mask = (temp/255.0)>0
					warp_cloth = np.array(Image.open(p.path_stage1_test+'warps/'+result_name.replace('.jpg','_cloth_bbkg.jpg')))/255.0
					return warp_cloth,warp_cloth_mask

			def f_cloth_mask():
				sname = trgt_line
				if(dataset=='mpv'):					
					seg = resize_img_forMPV(np.array(Image.open(sname.replace('all/','all_parsing/').replace('.jpg','.png'))))
				else:
					seg = np.array(Image.open(sname.replace('Img/img','cloth_parsing').replace('.jpg','.png')))
				cl_mask = (seg==5) + (seg==7) + (seg==6)
				return cl_mask

			if(stage==1):
				
				lm_line = base_name1
				#copy is very important becos dictionary gives pointers not the value
				if(woflm==0 and base_name1 in lm.keys() and base_name2 in lm.keys()):
					src_img_flm = lm[base_name1]['landmarks'].copy()
					trgt_img_flm = lm[base_name2]['landmarks'].copy()

					if(src_img_flm.shape[0]!=6 or trgt_img_flm.shape[0]!=6):
						print('flm error ')
						l_name.remove(l_name[-1])
						continue

					if(dataset=='mpv'):
						index_change=0
						src_img_flm[:,index_change] = (src_img_flm[:,index_change])+32
						trgt_img_flm[:,index_change] = (trgt_img_flm[:,index_change])+32

					src_img_flm = src_img_flm.reshape(-1)
					trgt_img_flm = trgt_img_flm.reshape(-1)
				else:
					src_img_flm = -1
					trgt_img_flm = -1

				src_img_hlm = get_hlm(src_line,dataset)
				src_img_hlm = src_img_hlm/255.0
				src_img_hlm = src_img_hlm.reshape(-1)

				trgt_img_hlm = get_hlm(trgt_line,dataset)
				trgt_img_hlm = trgt_img_hlm/255.0
				trgt_img_hlm = trgt_img_hlm.reshape(-1)
						

				if(src_img_hlm.shape[0]!=18 or trgt_img_hlm.shape[0]!=18):
					print('error hlm  ',src_line,trgt_line,src_img_hlm.shape,trgt_img_hlm.shape )
					continue
				
				l_src_img.append(src_img/255.0)
				l_trgt_img.append(trgt_img/255.0)

				l_src_hlm.append(src_img_hlm)
				l_src_flm.append(src_img_flm/255.0)
				
				l_trgt_hlm.append(trgt_img_hlm)			
				l_trgt_flm.append(trgt_img_flm/255.0)

				warp_cloth,warp_cloth_mask = f_warp_cloth(stage)
				l_warp_cloth.append(warp_cloth)
				l_warp_cloth_mask.append(warp_cloth_mask)			

			
			elif(stage==2):
				
				iuv,chk = f_iuv();
				l_iuv.append(iuv)
				warp_cloth,_ = f_warp_cloth(stage);
				l_warp_cloth.append(warp_cloth)

			elif(stage==3):
				
				iuv,chk = f_iuv();
				if(chk==-1):
					continue
				l_iuv.append(iuv)
				z = np.random.random((256,256,1)); l_z.append(z)
				
				cl_mask = f_cloth_mask()
				target_cl_mask = cl_mask.copy()				
				r = np.random.randint(3,5)
				kernel = np.ones((r,r), np.uint8)
				cl_mask = cv2.dilate(cl_mask.astype('float32'), kernel, iterations=1)>0
				
				iuv[:,:,:1] = (iuv[:,:,:1]*24.0).astype('uint8') ; 
				im_mask = (cl_mask + ((iuv[:,:,0]>=15) * (iuv[:,:,0]<=22)) + (iuv[:,:,0]==2) ).astype('float32')				
				im_mask = np.tile(im_mask[:,:,np.newaxis],(1,1,3))
				im=trgt_img
				im_masked = (im/255.0)*(1-im_mask)
				warp_cloth, warp_cloth_mask = f_warp_cloth(stage)				
				warp_cloth_temp=np.random.random(warp_cloth.shape)				
				warp_cloth = warp_cloth_temp*(1-warp_cloth_mask) + warp_cloth_mask*warp_cloth
				
				l_im_masked_wocl.append(rng11(im_masked))				
				im_masked = warp_cloth_mask*warp_cloth + (1-warp_cloth_mask)*im_masked
				l_im_masked.append(rng11(im_masked))
				l_warp_cloth.append(rng11(warp_cloth))
				
				if(womask==0):
					stage2_out=(np.array(Image.open(p.path_stage2_test+result_name)))					
					l_prev_cloth_mask.append((((stage2_out[:,:,np.newaxis])/255.0)>0.5).astype('float32'))
				else:
					l_prev_cloth_mask.append(warp_cloth_mask[:,:,:1])

		if(stage==1):
			
			yield( [l_src_hlm,l_trgt_hlm,l_src_flm],[l_warp_cloth,l_warp_cloth_mask],[l_src_img,l_trgt_img,l_name] )
		
		elif(stage==2):

			yield ([l_warp_cloth,l_iuv],[l_name])			

		elif(stage==3):

			yield([l_im_masked,l_warp_cloth,l_iuv,l_z,l_prev_cloth_mask],[l_name])








