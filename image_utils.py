'''
Author: Shuailin Chen
Created Date: 2021-05-27
Last Modified: 2021-05-30
	content: my image utilities
'''

import cv2
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  


from mylib import mathlib



def save_cv2_image_as_chinese_path(img, dst_path):
	''' using cv2 to saving image in chinese path

	Args:
		img (ndarray): image
		dst_path (str): destination path
	'''
	ext_name = osp.splitext(dst_path)[1]
	cv2.imencode(ext_name, img)[1].tofile(dst_path)


def read_cv2_image_as_chinese_path(img_path, dtype=np.uint8):
	''' using cv2 to read image in chinese path

	Args:
		img_path (str): image path
		dtype (np.dtype): image save data type. Default: np.uint8
	'''
	return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)


def save_image_by_cv2(img, dst_path):
	''' UNTESTED! Save image by cv2.imwrite, this function automatic transforms the data range and data type to adapt to cv2 

	Args:
		img (ndarray): image to be saved
		dst_path (str): save path

	Returns:
		True if succeed, False otherwise
	'''
	
	if img.dtype == np.uint8:
		new_img = img
	
	elif img.dtype in (np.float32, np.float64):
		
		# add a new axis for grayscale image
		if img.ndim==2:
			img = img[:, :, np.newaxis]

		new_img = np.empty_like(img, dtype=np.uint8)

		for ii in range(img.shape[2]):
			sub_img = img[..., ii]
			sub_img = mathlib.min_max_map(sub_img)
			sub_img = (255*sub_img).astype(np.uint8)
			new_img[..., ii] = sub_img

	new_img = new_img.squeeze()
	return save_cv2_image_as_chinese_path(new_img, dst_path)


def plot_surface(img, cmap='jet'):
	''' plot 3D surface of image
	'''

	h, w = img.shape

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	X = np.arange(h)
	Y = np.arange(w)
	X, Y = np.meshgrid(X, Y)

	surf = ax.plot_surface(X, Y, img, cmap=cmap)

	return fig