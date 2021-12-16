from keras.applications.vgg16 import VGG16
from keras import Model
from keras.layers import *
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error
import models.my_layers as my_layers
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


def get_model():

	hlm1  = Input(shape=(18,))
	hlm2  = Input(shape=(18,))
	flm1  = Input(shape=(12,))

	hlm  = Input(shape=(18,))

	node=18
	f1 = hlm1
	f2 = hlm2

	ip = my_layers.FeatureCorrelation(out_dim = node**2)([f1,f2])

	x = Concatenate(axis=-1)([hlm1,hlm2,ip,flm1])
	x = Dense(900,activation = 'relu')(x)
	x = InstanceNormalization()(x)
	x = Dense(800,activation = 'relu')(x)
	x = InstanceNormalization()(x)
	x = Dense(600,activation = 'relu')(x)
	x = InstanceNormalization()(x)
	x = Dense(500,activation = 'relu')(x)
	x = InstanceNormalization()(x)
	x = Dense(250,activation = 'relu')(x)
	x = InstanceNormalization()(x)
	x = Dense(100,activation = 'relu')(x)
	x = InstanceNormalization()(x)
	flm2 = Dense(12,activation = 'sigmoid')(x)

	model_final = Model(inputs = [hlm1,hlm2,flm1] , outputs = [flm2])

	hlm_1  = Input(shape=(18,))
	hlm_2  = Input(shape=(18,))
	flm_1  = Input(shape=(12,))

	flm_2 = model_final([hlm_1,hlm_2,flm_1])
	flm_11 = model_final([hlm_2,hlm_1,flm_2])


	model_rest = Model(inputs = [hlm_1,hlm_2,flm_1],outputs = [flm_2,flm_11])


	rms = Adam(lr=0.001)#RMSprop(lr=5e-4)
	model_rest.compile(optimizer=rms, loss=[mean_squared_error,mean_squared_error], metrics=["accuracy","accuracy"])
	return model_rest,model_final


# model_rest,model_final = get_model()
#3
def get_model_v3(compile_model=True):

	hlm1  = Input(shape=(18,))
	hlm2  = Input(shape=(18,))
	flm1  = Input(shape=(12,))

	hlm  = Input(shape=(18,))

	node=18
	f1 = hlm1
	f2 = hlm2

	ip = my_layers.FeatureCorrelation(out_dim = node**2)([f1,f2])

	x = Concatenate(axis=-1)([hlm1,hlm2,ip,flm1])
	x = Concatenate(axis=-1)([hlm1,hlm2,flm1])
	x = Dense(900,activation = 'relu')(x)
	x = InstanceNormalization()(x)
	x = Dense(800,activation = 'relu')(x)
	x = InstanceNormalization()(x)
	x = Dense(600,activation = 'relu')(x)
	x = InstanceNormalization()(x)
	x = Dense(500,activation = 'relu')(x)
	x = InstanceNormalization()(x)
	x = Dense(250,activation = 'relu')(x)
	x = InstanceNormalization()(x)
	x = Dense(100,activation = 'relu')(x)
	flm2 = Dense(12,activation = 'sigmoid')(x)

	model_final = Model(inputs = [hlm1,hlm2,flm1] , outputs = [flm2])

	hlm_1  = Input(shape=(18,))
	hlm_2  = Input(shape=(18,))
	flm_1  = Input(shape=(12,))

	flm_2 = model_final([hlm_1,hlm_2,flm_1])
	# # flm_11 = model_final([hlm_2,hlm_1,flm_2])


	# # model_rest = Model(inputs = [hlm_1,hlm_2,flm_1],outputs = [flm_2,flm_11])
	model_rest = Model(inputs = [hlm_1,hlm_2,flm_1],outputs = [flm_2])


	rms = Adam(lr=0.0001)#RMSprop(lr=5e-4)
	if(compile_model==True):
		model_rest.compile(optimizer=rms, loss=[mean_squared_error], metrics=["accuracy"])
	return model_rest,model_final