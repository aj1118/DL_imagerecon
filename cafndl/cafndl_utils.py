# modules
import numpy as np
def augment_data(data_xy, axis_xy=[1,2], augment={'flipxy':0,'flipx':0,'flipy':0}):
	if 'flipxy' in augment and augment['flipxy']:
		data_xy = np.swapaxes(data_xy, axis_xy[0], axis_xy[1])
	if 'flipx' in augment and augment['flipx']:
		if axis_xy[0] == 0:
			data_xy = data_xy[::-1,...]
		if axis_xy[0] == 1:
			data_xy = data_xy[:, ::-1,...]
	if 'flipy' in augment and augment['flipy']:
		if axis_xy[1] == 1:
			data_xy = data_xy[:, ::-1,...]
		if axis_xy[1] == 2:
			data_xy = data_xy[:, :, ::-1,...]
	if 'shiftx' in augment and augment['shiftx']>0:
		if axis_xy[0] == 0:
			data_xy[:-augment['shiftx'],...] = data_xy[augment['shiftx']:,...]
		if axis_xy[0] == 1:
			data_xy[:,:-augment['shiftx'],...] = data_xy[:,augment['shiftx']:,...]
	if 'shifty' in augment and augment['shifty']>0:
		if axis_xy[1] == 1:
			data_xy[:,:-augment['shifty'],...] = data_xy[:,augment['shifty']:,...]
		if axis_xy[1] == 2:
			data_xy[:,:,:-augment['shifty'],...] = data_xy[:,:,augment['shifty']:,...]
	return data_xy
