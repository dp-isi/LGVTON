from models.hourglass_stage3.hg_blocks import create_hourglass_network, euclidean_loss, bottleneck_block, bottleneck_mobile
from keras.layers import *
import keras.backend as K
from keras.models import Model
from keras.initializers import RandomNormal
from keras import optimizers
from keras_contrib.losses import DSSIMObjective
from keras.activations import relu
import tensorflow as tf
import numpy as np


# -------------------------------------------------------
gan_loss='mse'

def build_generator():
	
	gen_model = create_hourglass_network(num_classes = 3, num_stacks = 1, num_channels = 128,\
	 									inres = (256,256), outres = (256,256), bottleneck = bottleneck_mobile,inchannel = 8)
	return gen_model

def build_discriminator_patchgan(df=12):

    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img_shape=(256,256)
    gen_out = Input(shape=img_shape+(3,))
    warp_cloth = Input(shape=img_shape+(3,))
    combined_imgs = Concatenate(axis=-1)([gen_out, warp_cloth])

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same',activation='sigmoid')(d4)
    model = Model([gen_out,warp_cloth], [validity])
    model.compile(loss=['mse'], optimizer=optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.999))#optimizer='sgd')

    return model


from keras.applications.vgg19 import VGG19

feature_layers = ['block1_conv1', 'block2_conv1','block3_conv1', 'block4_conv1','block5_conv1']
fmaps = [(224,224,64),(112,112,128),(56,56,256),(28,28,512),(14,14,512)]
w =[]
for i in range(5):
	w.append(1.0/(np.prod(fmaps[i])))

def custom_loss(y_true,y_pred):

	im=(y_pred+1)/2.0
	im_gt=(y_true+1)/2.0

	vgg19=VGG19(include_top=False)

	outer_model = Model(inputs=[(vgg19).layers[0].input],outputs=[  (vgg19).get_layer(feature_layers[0]).output,\
		(vgg19).get_layer(feature_layers[1]).output,\
		(vgg19).get_layer(feature_layers[2]).output,\
		(vgg19).get_layer(feature_layers[3]).output,\
		(vgg19).get_layer(feature_layers[4]).output  ]  )

	im = outer_model([tf.image.resize(im, (224,224))])
	im_gt1 = outer_model([tf.image.resize(im_gt, (224,224))])

	pdist = w[0]*K.mean(K.square(im[0]-im_gt1[0])) +\
	w[1]*K.mean(K.square(im[1]-im_gt1[1])) + \
	w[2]*K.mean(K.square(im[2]-im_gt1[2])) +\
	w[3]*K.mean(K.square(im[3]-im_gt1[3])) \
	+ w[4]*K.mean(K.square(im[4]-im_gt1[4]))
	
	lossval = 1000*pdist + DSSIMObjective(kernel_size=5)(y_true,y_pred)

	return lossval

def build_gan(test=0):
	
	img_shape=(256,256)

	iuv = Input(shape=img_shape+(3,))    
	warp_cloth = Input(shape=img_shape+(3,))
	im_masked = Input(shape=img_shape+(3,))
	prev_cloth_mask = Input(shape=img_shape+(1,))
	z = Input(shape=img_shape+(1,))

	generator = build_generator()
	discriminator = build_discriminator_patchgan()

	gen_ip = Concatenate()([im_masked,iuv,prev_cloth_mask,z])
	gen_out = generator([gen_ip])
	
	gen_out_img = Lambda(lambda a: a[:,:,:,:3])(gen_out)
	gen_out_mask = Lambda(lambda a: a[:,:,:,3:4])(gen_out)
	convexcombi_gen = Lambda(lambda x: x[:,:,:,-3:])(gen_out) 
	
	iplist = [ im_masked,iuv,prev_cloth_mask,warp_cloth,z]

	generator_cnn = Model( inputs=iplist, outputs=[ convexcombi_gen ] )

	discriminator.trainable=False
	dis_out = discriminator([convexcombi_gen,warp_cloth])

	gan = Model(inputs=iplist, outputs=[ convexcombi_gen, dis_out] )	
	
	gan.compile(loss = [custom_loss,'mse'] , loss_weights=[1,1],optimizer=optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.999))

	return generator_cnn,gan, discriminator
