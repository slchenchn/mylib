'''
Author: Shuailin Chen
Created Date: 2021-05-24
Last Modified: 2021-05-25
	content: simulate data with Wishart noise
'''

import os.path as osp

import cv2
import numpy as np

from mylib import polSAR_utils as psr


def generate_Wishart_noise_from_img(img, ENL):
	''' Generate Wishart noise from a image

	Args:
		img (ndarray): image data, in shape of [H, W, C] 
		ENL (int): equivalent number of looks

	Returns:
		tmp (ndarray): reshape of the original image, in shape of [9, H, W]
		noise (ndarray): complex-valued noise image in shape of [9, H, W]	
	'''

	h, w, c = img.shape

	assert c == 3

	# change to PolSAR's data format
	img = img.reshape(-1, 3).transpose()
	tmp = np.empty((3, *img.shape), dtype=img.dtype)
	tmp[[0, 1, 2], [0, 1, 2], :] = img[:, :]

	noise = psr.wishart_noise(tmp, ENL=ENL)

	return tmp.reshape(9, h , w), noise.reshape(9, h, w)


def generate_Wishart_noise_from_PolSAR_data(img, ENL):
	''' Generate Wishart noise from a PolSAR data

	Args:
		img (ndarray): PolSAR data, in shape of [C, H, W] 
		ENL (int): equivalent number of looks

	Returns:
		noise (ndarray): complex-valued noise image in shape of [9, H, W]	
	'''

	img = psr.as_format(img, 'complex_vector_9')
	c, h, w = img.shape

	# change to PolSAR's data format
	img = img.reshape(3, 3, -1)

	noise = psr.wishart_noise(img, ENL=ENL)

	return noise.reshape(9, h, w)


if __name__ == '__main__':
	_tmp_path = r'/home/csl/code/PolSAR_N2N/tmp'
	path = r'/home/csl/code/PolSAR_N2N/data/BSR/BSDS500/data/images/train/25098.jpg'
	img = cv2.imread(path)
	# img = cv2.imread(r'a2.jpg')
	# img = cv2.imread(r'test.jpg')
	clean, noise = generate_Wishart_noise_from_img(img, 4)
	pauli = psr.rgb_by_c3(noise, type='sinclair')
	cv2.imwrite(osp.join(_tmp_path, 'pauli.png'), (255*pauli).astype(np.uint8))

	# C3_path = r'E:\code\python\深度学习滤波\codes(主要代码)\实验\noiseSynthetic\tmp'
	# img = psr.read_c3(C3_path)
	# noise = generate_Wishart_noise_from_PolSAR_data(img, 4)
	# pauli_ori = psr.rgb_by_c3(img)
	# pauli_noise = psr.rgb_by_c3(noise)
	# cv2.imwrite('pauli_ori.png', cv2.cvtColor((255*pauli_ori).astype(np.uint8), cv2.COLOR_RGB2BGR))
	# cv2.imwrite('pauli_noise.png', cv2.cvtColor((255*pauli_noise).astype(np.uint8), cv2.COLOR_RGB2BGR))
