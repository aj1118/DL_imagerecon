from scipy import io as sio
import numpy as np
import os
import dicom
import nibabel as nib
import datetime
from cafndl_fileio import *
from cafndl_utils import *
from cafndl_network import *
from cafndl_metrics import *
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

'''
convert dicom to nifti
$ mkdir DRF100_nifti
$ dicom2nifti DRF100 DRF100_nifti
'''


'''
dataset
'''
filename_checkpoint = '../ckpt/model_mc.ckpt'
list_dataset_test =  [
				{
				 'inputs':['/research_data/AJ/data_lowdoseCT/trial01/DRF020_nifti/802_.nii.gz',
				 			'/research_data/AJ/data_lowdoseCT/trial01/DRF100_nifti/803_.nii.gz'],
				 'gt':'/research_data/AJ/data_lowdoseCT/trial01/DRF001_nifti/800_.nii.gz'
				},
				{
				 'inputs':['/research_data/AJ/data_lowdoseCT/trial00/DRF020_nifti/802_.nii.gz',
				 			'/research_data/AJ/data_lowdoseCT/trial00/DRF100_nifti/803_.nii.gz'],
				 'gt':'/research_data/AJ/data_lowdoseCT/trial00/DRF001_nifti/802_.nii.gz'
				}
				]
filename_results = '../results/012018'
num_dataset_test = len(list_dataset_test)
print('process {0} data description'.format(num_dataset_test))

'''
augmentation
'''
list_augments = []

'''
generate test data
'''
list_test_input = []
list_test_gt = []
for index_data in range(num_dataset_test):
	# directory
	list_data_test_input = []
	for path_test_input in list_dataset_test[index_data]['inputs']:
		# load data
		data_test_input = prepare_data_from_nifti(path_test_input, list_augments)
		list_data_test_input.append(data_test_input)
	data_test_input = np.concatenate(list_dataset_test_input, axis=-1)

	# get ground truth
	path_test_gt = list_dataset_test[index_data]['gt']
	data_test_gt = prepare_data_from_nifti(path_test_gt, [])

	# append
	list_test_input.append(data_test_input)
	list_test_gt.append(data_test_gt)

# generate test dataset
scale_data = 100.
data_test_input = scale_data * np.concatenate(list_test_input, axis = 0)
data_test_gt = scale_data * np.concatenate(list_test_gt, axis = 0)
data_test_residual = data_test_gt - data_test_input[:,:,:,0:1]
print('mean, min, max')
print(np.mean(data_test_input.flatten()),np.min(data_test_input.flatten()),np.max(data_test_input.flatten()))
print(np.mean(data_test_gt.flatten()),np.min(data_test_gt.flatten()),np.max(data_test_gt.flatten()))
print(np.mean(data_test_residual.flatten()),np.min(data_test_residual.flatten()),np.max(data_test_residual.flatten()))
print('generate test dataset with augmentation size {0},{1}'.format(
	data_test_input.shape, data_test_gt.shape))

'''
setup parameters
'''
keras_memory = 0.3
keras_backend = 'tf'
num_channel_input = data_test_input.shape[-1]
num_channel_output = data_test_gt.shape[-1]
img_rows = data_test_input.shape[1]
img_cols = data_test_gt.shape[1]
num_poolings = 3
num_conv_per_pooling = 3
lr_init = 0.001
with_batch_norm = True
ratio_validation = 0.1
batch_size = 4
always_retrain = 1
num_epoch = 100
print('setup parameters')


'''
define model
'''
setKerasMemory(keras_memory)
model = deepEncoderDecoder(num_channel_input = num_channel_input,
						num_channel_output = num_channel_output,
						img_rows = img_rows,
						img_cols = img_cols,
						lr_init = lr_init,
						num_poolings = num_poolings,
						num_conv_per_pooling = num_conv_per_pooling,
						with_bn = with_batch_norm, verbose=1)
print('train model:', filename_checkpoint)
print('parameter count:', model.count_params())


'''
load network
'''
model.load_weights(filename_checkpoint)
print('model load from' + filename_checkpoint)

'''
apply model
'''
t_start_pred = datetime.datetime.now()
data_test_output = model.predict(data_test_input, batch_size=batch_size)
# clamp
clamp_min = -0.5
clamp_max = 0.5
data_test_output = np.maximum(np.minimum(data_test_output, clamp_max), clamp_min)

# add
data_test_output += data_test_input[:,:,:,0:1]

t_end_pred = datetime.datetime.now()
print('predict on data size {0} using time {1}'.format(
	data_test_output.shape, t_end_pred - t_start_pred))

'''
export images
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
num_sample = data_test_output.shape[0]
list_err_pred = []
list_err_input = []
aug_err = 10
for i in range(num_sample):
	# get image
	im_gt = np.squeeze(data_test_gt[i,:,:,0]).T
	im_input = np.squeeze(data_test_input[i,:,:,0]).T
	im_pred = np.squeeze(data_test_output[i,:,:,0]).T

	# get error
	err_pred = getErrorMetrics(im_pred, im_gt)
	err_input = getErrorMetrics(im_input, im_gt)
	list_err_pred.append(err_pred)
	list_err_input.append(err_input)

	# display
	im_toshow = [im_gt, im_input, (im_input-im_gt)*aug_err, im_pred, (im_pred-im_gt)*aug_err]
	im_toshow = np.abs(np.concatenate(im_toshow, axis=1))
	plt.figure(figsize=[20,8])
	plt.imshow(im_toshow, clim=[0,0.5], cmap='gray')
	im_title = 'sample #{0}, input PSNR {1:.4f}, SSIM {2:.4f}, predict PSNR {3:.4f}, SSIM {4:.4f}'.format(
				i, err_input['psnr'], err_input['ssim'], err_pred['psnr'], err_pred['ssim'])
	plt.title(im_title)
	print(im_title)
	path_figure = filename_results+'_{0}.png'.format(i)
	plt.savefig(path_figure)

'''
export results
'''
print('input average metrics:', {k:np.mean([x[k] for x in list_err_input]) for k in list_err_input[0].keys()})
print('prediction average metrics:', {k:np.mean([x[k] for x in list_err_pred]) for k in list_err_pred[0].keys()})# save history dictionary
import json
result_error = {'err_input':list_err_input,
				'err_pred':list_err_pred}
path_error = filename_results+'_error.json'
with open(path_error, 'w') as outfile:
    json.dump(result_error, outfile)
print('error exported to {0}'.format(path_error))
