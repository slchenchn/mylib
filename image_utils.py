'''
Author: Shuailin Chen
Created Date: 2021-05-27
Last Modified: 2021-05-27
	content: my image utilities
'''

import cv2
import numpy as np
import os.path as osp


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