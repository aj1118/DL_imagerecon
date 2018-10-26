## AJ Kruse Simple CNN for image recon
## 18-15-01 Update: testing optimal infrastructure

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, merge, Conv2D, Conv2DTranspose, BatchNormalization, Convolution2D, MaxPooling2D, UpSampling2D, Dense, concatenate
from keras.layers.merge import add as keras_add
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.losses import mean_absolute_error, mean_squared_error
from keras import backend as K
from cafndl_metrics import PSNRLoss
import numpy as np

# clean up
def clearKerasMemory():
    K.clear_session()

# use part of memory
def setKerasMemory(limit=0.3):
    from tensorflow import ConfigProto as tf_ConfigProto
    from tensorflow import Session as tf_Session
    from keras.backend.tensorflow_backend import set_session
    config = tf_ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = limit
    set_session(tf_Session(config=config))

# encoder-deocder
def deepEncoderDecoder(num_channel_input=1, num_channel_output=1,
	img_rows=128, img_cols=128, y=np.array([-1,1]),
	lr_init=None, loss_function=mean_absolute_error, metrics_monitor=[PSNRLoss, mean_absolute_error, mean_squared_error],
	num_poolings = 3, num_conv_per_pooling = 3,
	with_bn=False, verbose=1):
	# BatchNorm
	if with_bn:
		lambda_bn = lambda x: BatchNormalization()(x)
	else:
		lambda_bn = lambda x: x
	# layers
#     For 2D data (e.g. image), "tf" assumes (rows, cols, channels) while "th" assumes (channels, rows, cols).
	inputs = Input((img_rows, img_cols, num_channel_input))
	if verbose:
		print(inputs)

	#step1
	conv1 = inputs
	num_channel_first = 32
	for i in xrange(num_conv_per_pooling):
		conv1 = Conv2D(num_channel_first, (3, 3), padding="same", activation="relu")(conv1)
		conv1 = lambda_bn(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	if verbose:
		print(conv1,pool1)

	# encoder pools
	convs = [inputs, conv1]
	pools = [inputs, pool1]
	list_num_features = [num_channel_input, num_channel_first]
	for i in xrange(1, num_poolings):
		#step2
		conv_encoder = pools[-1]
		num_channel = num_channel_first*(2**(i-1))
		for j in xrange(num_conv_per_pooling):
			conv_encoder = Conv2D(num_channel, (3, 3), padding="same", activation="relu")(conv_encoder)
			conv_encoder = lambda_bn(conv_encoder)
		pool_encoder = MaxPooling2D(pool_size=(2, 2))(conv_encoder)
		if verbose:
			print(conv_encoder,pool_encoder)
		pools.append(pool_encoder)
		convs.append(conv_encoder)
		list_num_features.append(num_channel)

	# center connection
	conv_center = Conv2D(list_num_features[-1], (3, 3), padding="same", activation="relu",
					   kernel_initializer='zeros',
					   bias_initializer='zeros')(pools[-1])
	conv_center = keras_add([pools[-1], conv_center])
	conv_decoders = [conv_center]
	if verbose:
		print(conv_center)

	# decoder steps
	for i in xrange(1, num_poolings+1):
# 		print('decoder', i, convs, pools)
# 		print(UpSampling2D(size=(2, 2))(conv_center))
# 		print(convs[-i])
		up_decoder = concatenate([UpSampling2D(size=(2, 2))(conv_decoders[-1]), convs[-i]])
		conv_decoder = up_decoder
		for j in xrange(num_conv_per_pooling):
			conv_decoder = Conv2D(list_num_features[-i], (3, 3), padding="same", activation="relu")(conv_decoder)
			conv_decoder = lambda_bn(conv_decoder)
		conv_decoders.append(conv_decoder)
		if verbose:
			print(conv_decoder,up_decoder)

	# output layer
	conv_decoder = conv_decoders[-1]
	if max(abs(y))<=1:
		if min(np.array(y).flatten())<0:
#         tanh -1~+1
			conv_output = Conv2D(num_channel_output, (1, 1), padding="same", activation="tanh")(conv_decoder)
			print('use tanh activation')
		else:
			conv_output = Conv2D(num_channel_output, (1, 1), padding="same", activation='sigmoid')(conv_decoder)
			print('use sigmoid activation')
	else:
		conv_output = Conv2D(num_channel_output, (1, 1), padding="same", activation='linear')(conv_decoder)
		print('use linear activation')
	if verbose:
		print(conv_output)

	# model
	model = Model(outputs=conv_output, inputs=inputs)
	if verbose:
		print(model)

	# fit...Adam
	if lr_init is not None:
		optimizer = Adam(lr=lr_init) #,0.001 rho=0.9, epsilon=1e-08, decay=0.0)
	else:
		optimizer = Adam()
	model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics_monitor)

	return model
