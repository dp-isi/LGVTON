import os 
import tensorflow.keras as keras
from keras import backend as K
from keras.layers import Layer
from keras.models import Model as models
import keras
from keras_applications.resnext import ResNeXt101
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from keras.models import Sequential
import keras.layers as kl
from keras.activations import linear
from keras.models import Model

import sys
class FeatureL2Norm(Layer):
	def __init__(self):
		super(FeatureL2Norm, self).__init__()

	def call(self, feature):
		epsilon = 1e-6  #sys.float_info.epsilon#
		#print(feature.size())
		#print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
		norm = tf.math.pow(tf.math.reduce_sum(tf.math.pow(feature,2),1)+epsilon,0.5)
		norm = K.repeat_elements(K.expand_dims(norm,1),feature.shape[1],axis=1)
		return tf.math.divide(feature,norm)

class FeatureCorrelation(Layer):
	def __init__(self,out_dim,**kwargs):
		super(FeatureCorrelation, self).__init__(**kwargs)
		self.out_dim=out_dim
		
	
	def call(self,feature):
		# feature_A = tf.convert_to_tensor(np.zeros(shape=(10,4096)))
		# feature_B = tf.convert_to_tensor(np.zeros(shape=(10,4096)))
		[feature_A,feature_B] = feature
		b,a = feature_A.get_shape().as_list()

		feature_A = tf.reshape(feature_A,(-1,a,1))
		feature_B = tf.reshape(feature_B,(-1,a,1))
		feature_B = tf.transpose(feature_B,perm=(0,2,1))

		feature_mul = K.batch_dot(feature_A,feature_B)
		correlation_tensor = tf.reshape(feature_mul,(-1,a*a))

		return correlation_tensor

	def compute_output_shape(self, input_shape):
		return (input_shape[0][0],self.out_dim)

	def get_config(self):
		config = super(FeatureCorrelation, self).get_config()
		config.update({'out_dim': self.out_dim})
		# return ({'out_dim': self.out_dim})
		return config

	# def from_config(cls, config):
	# 	return cls(**config)

		






