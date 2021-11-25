from models.hourglass_stage2.hg_blocks import create_hourglass_network, euclidean_loss, bottleneck_block, bottleneck_mobile
from keras.layers import *
from keras.models import Model
from keras.initializers import RandomNormal
from keras import optimizers
from keras_contrib.losses import DSSIMObjective
from keras.activations import relu
import tensorflow as tf
def build_generator():
	
	gen_model = create_hourglass_network(num_classes = 1, num_stacks = 1, num_channels = 128,\
	 									inres = (256,256), outres = (256,256), bottleneck = bottleneck_mobile,inchannel = 6)
	return gen_model


from keras.losses import binary_crossentropy
def build_gan():

	img_shape=(256,256)

	iuv = Input(shape=img_shape+(3,))    
	warp_cloth = Input(shape=img_shape+(3,))

	generator = build_generator()
	
	gen_ip = Concatenate()([warp_cloth,iuv])
	gen_out = generator([gen_ip])
	gen_out1 = Lambda( lambda d:  d[:,:,:,0]  )(gen_out) #[d[:,:,:,:3],d[:,:,:,3]


	generator_cnn = Model( inputs=[warp_cloth,iuv], outputs=[ gen_out1 ] )

	generator_cnn.compile(loss = ['binary_crossentropy'] , optimizer=optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999))

	return generator_cnn
