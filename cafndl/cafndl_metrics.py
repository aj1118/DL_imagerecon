import numpy as np
# use skimage metrics
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim

# psnr with TF
try:
	from keras import backend as K
	from tensorflow import log as tf_log
	from tensorflow import constant as tf_constant
	import tensorflow as tf
except:
	print('import keras and tf backend failed')

def PSNRLoss(y_true, y_pred):
	"""
	PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

	It can be calculated as
	PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

	When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
	However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
	Thus we remove that component completely and only compute the remaining MSE component.
	"""
	try:
		#use theano
		return 20.*np.log10(K.max(y_true)) -10. * np.log10(K.mean(K.square(y_pred - y_true)))
	except:
		denominator = tf_log(tf_constant(10.0))
		return 20.*tf_log(K.max(y_true)) / denominator -10. * tf_log(K.mean(K.square(y_pred - y_true))) / denominator
	return 0

#get error metrics, for psnr, ssimr, rmse, score_ismrm
def getErrorMetrics(im_pred, im_gt, mask = None):
	# flatten array
	im_pred = np.array(im_pred).astype(np.float).flatten()
	im_gt = np.array(im_gt).astype(np.float).flatten()
	if mask is not None:
		mask = np.array(mask).astype(np.float).flatten()
		im_pred = im_pred[mask>0]
		im_gt = im_gt[mask>0]
	mask=np.abs(im_gt.flatten())>0

	# check dimension
	assert(im_pred.flatten().shape==im_gt.flatten().shape)

	# NRMSE
	rmse_pred = compare_nrmse(im_gt, im_pred)

	# PSNR
	try:
		psnr_pred = compare_psnr(im_gt, im_pred)
	except:
		psnr_pred = psnr(im_gt, im_pred)
		print('use psnr')
	
	# ssim
	ssim_pred = compare_ssim(im_gt, im_pred)
	score_ismrm = sum((np.abs(im_gt.flatten()-im_pred.flatten())<0.1)*mask)/(sum(mask)+0.0)*10000

	return {'rmse':rmse_pred,'psnr':psnr_pred,'ssim':ssim_pred,'score_ismrm':score_ismrm}
	