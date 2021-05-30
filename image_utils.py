'''
Author: Shuailin Chen
Created Date: 2021-05-27
Last Modified: 2021-05-30
	content: my image utilities
'''

import cv2
import numpy as np
import os.path as osp

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
	''' Save image by cv2.imwrite, this function automatic transforms the data range and data type to adapt to cv2 

	Args:
		img (ndarray): image to be saved
		dst_path (str): save path

	Returns:
		True if succeed, False otherwise
	'''
	
	if img.dtype == np.uint8:
		new_img = img
	
	elif img.dtype in (np.float32, np.float64):
		new_img = np.empty_like(img, dtype=np.uint8)
		
		# add a new axis for grayscale image
		if img.ndim==2:
			img = img[:, :, np.newaxis]

		for ii in range(img.shape[2]):
			sub_img = img[..., ii]
			sub_img = mathlib.min_max_map(sub_img)
			sub_img = (255*sub_img).astype(np.uint8)
			new_img[..., ii] = sub_img

	return save_cv2_image_as_chinese_path(new_img, dst_path)