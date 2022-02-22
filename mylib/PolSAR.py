'''
Author: Shuailin Chen
Created Date: 2021-05-13
Last Modified: 2022-02-21
	content: advanced version of polSAR_utils.py, written in objection-oriented style, 
    NOTE: **undone**.
'''

"""
import os
import os.path as osp
import math

import numpy as np 
from matplotlib import pyplot as plt 
import cv2
import bisect
from numpy import ndarray
import torch

from mylib import file_utils as fu
from typing import Union
from mylib import polSAR_utils as psr
from mylib import mathlib

BIN_FIELS = {
        'C3': ['C11.bin', 'C12_real.bin', 'C12_imag.bin',    
            'C13_real.bin', 'C13_imag.bin', 'C22.bin', 'C23_real.bin', 
            'C23_imag.bin', 'C33.bin',],
        'T3': ['T11.bin', 'T12_real.bin', 'T12_imag.bin', 
            'T13_real.bin', 'T13_imag.bin', 'T22.bin', 'T23_real.bin', 
            'T23_imag.bin', 'T33.bin',],
        's2': ['s11.bin', 's12.bin', 's21.bin', 's22.bin']
        }


class PolSAR():
    ''' PolSAR data
    Only accept .bin file with float data type, 'bsq' interleave 
    ATTENTION: Untested
    '''
    def __init__(self, path, data_format=None) -> None:
        self.path = path
        self.data_format = self._get_data_format(data_format)
        self.shape = self._get_shape()
        self.data = None
        self.storage_mode = None

    def read_data(self, is_print=True, storage_mode='save_space'):
        ''' Read PolSAR data 

        Args:
            is_print (bool): if to print related infos. Default: True
            storage_mode (str): for C3 data, 'save_space' or
                'complex_vector_6' or 'complex_vector_9'. Default:
                'save_space'
        '''

        if is_print:
            print(f'reading {self.data_format} data from {self.path}')

        if self.data_format=='s2':
            self.data = self._read_s2()
        elif self.data_format=='C3':
            self.data = self._read_c3()
        else:
            raise NotImplementedError

    def as_storage_mode(self, target_storage_mode, inplace=True):
        ''' Change the data storage mode of C3 or T3 data
        
        Args:
            storage_mode (str): 'save_space' or 
                'complex_vector_6' or 'complex_vector_9'. Default: 
                'save_space'
            inplace (bool): if to change self.data's storage mode
        '''

        assert self.data_format in ('C3', 'T3')
        if self.storage_mode == target_storage_mode:
            return

        # complex vector 6
        if self.storage_mode == 'complex_vector_6':
            if target_storage_mode == 'save_space':
                c11 = self.data[0, :, :].real
                c12_real = self.data[1, :, :].real
                c12_img = self.data[1, :, :].imag
                c13_real = self.data[2, :, :].real
                c13_img = self.data[2, :, :].imag
                c22 = self.data[3, :, :].real
                c23_real = self.data[4, :, :].real
                c23_img = self.data[4, :, :].imag
                c33 = self.data[5, :, :].real
                c3 = np.stack((c11, c12_real, c12_img, c13_real, c13_img, c22, c23_real, c23_img, c33), axis=0)
            elif target_storage_mode == 'complex_vector_9':
                c11 = self.data[0, :, :]
                c12 = self.data[1, :, :]
                c13 = self.data[2, :, :]
                c22 = self.data[3, :, :]
                c23 = self.data[4, :, :]
                c33 = self.data[5, :, :]
                c3 = np.stack((c11, c12, c13, c12.conj(), c22, c23, c13.conj(), c23.conj(), c33), axis=0)
            else:
                raise LookupError('wrong storage mode')

        # complex_vector_9
        elif self.storage_mode == 'complex_vector_9': 
            if target_storage_mode == 'save_space':
                c11 = self.data[0, :, :].real
                c12_real = self.data[1, :, :].real
                c12_img = self.data[1, :, :].imag
                c13_real = self.data[2, :, :].real
                c13_img = self.data[2, :, :].imag
                c22 = self.data[4, :, :].real
                c23_real = self.data[5, :, :].real
                c23_img = self.data[5, :, :].imag
                c33 = self.data[8, :, :].real
                c3 = np.stack((c11, c12_real, c12_img, c13_real, c13_img, c22, c23_real, c23_img, c33), axis=0)
            elif target_storage_mode == 'complex_vector_6':
                c11 = self.data[0, :, :]
                c12 = self.data[1, :, :]
                c13 = self.data[2, :, :]
                c22 = self.data[4, :, :]
                c23 = self.data[5, :, :]
                c33 = self.data[8, :, :]
                c3 = np.stack((c11, c12, c13, c22, c23, c33), axis=0)
            else:
                raise LookupError('wrong out format')
        
        # save_space
        elif self.storage_mode == 'save_space' :
            if target_storage_mode == 'complex_vector_9':
                c11 = self.data[0, :, :]
                c12 = self.data[1, :, :] + 1j*self.data[2, :, :]
                c13 = self.data[3, :, :] + 1j*self.data[4, :, :]
                c22 = self.data[5, :, :]
                c23 = self.data[6, :, :] + 1j*self.data[7, :, :]
                c33 = self.data[8, :, :]
                c3 = np.stack((c11, c12, c13, c12.conj(), c22, c23, c13.conj(), c23.conj(), c33), axis=0)
            elif target_storage_mode == 'complex_vector_6':
                c11 = self.data[0, :, :]
                c12 = self.data[1, :, :] + 1j*self.data[2, :, :]
                c13 = self.data[3, :, :] + 1j*self.data[4, :, :]
                c22 = self.data[5, :, :]
                c23 = self.data[6, :, :] + 1j*self.data[7, :, :]
                c33 = self.data[8, :, :]
                c3 = np.stack((c11, c12, c13,  c22, c23,  c33), axis=0)
            else:
                raise LookupError('wrong out format')
        
        if inplace:
            self.storage_mode = target_storage_mode
            self.data = c3
        return c3

    def to_rgb(self, mode='pauli'):
        ''' False color composition

        Args:
            mode (str): 'pauli' or 'sinclair'. Default: 'pauli'
        '''

        # false color compose
        if mode=='pauli':
            if self.data_format == 'C3':
                c3 = self.as_storage_mode('complex_vector_6', inplace=False)
                R = 0.5*(c3[0, :, :]+c3[5, :, :])-c3[2, :, :]
                G = self.data[3, :, :]
                B = 0.5*(c3[0, :, :]+c3[5, :, :])+c3[2, :, :]
            elif self.data_format == 'T3':
                t3 = self.as_storage_mode('complex_vector_6', inplace=False)
                R = t3[3, :, :]
                G = t3[5, :, :]
                B = t3[0, :, :]
            elif self.data_format == 's2':
                s11 = self.data[0, :, :]
                s12 = self.data[1, :, :]
                s21 = self.data[2, :, :]
                s22 = self.data[3, :, :]
                R=0.5*np.conj(s11-s22)*(s11-s22)
                G=0.5*np.conj(s12+s21)*(s12+s21)
                B=0.5*np.conj(s11+s22)*(s11+s22)
        else:
            raise NotImplementedError

        # clip
        R = np.abs(R)
        G = np.abs(G)
        B = np.abs(B)
        R[R<np.finfo(float).eps] = np.finfo(float).eps
        G[G<np.finfo(float).eps] = np.finfo(float).eps
        B[B<np.finfo(float).eps] = np.finfo(float).eps

        # logarithm 
        R = 10*np.log10(R)
        G = 10*np.log10(G)
        B = 10*np.log10(B)
        
        # normalize
        R = mathlib.min_max_contrast_median_map(R)
        G = mathlib.min_max_contrast_median_map(G)
        B = mathlib.min_max_contrast_median_map(B)

        rgb = np.stack((R, G, B), axis=2)
        return (rgb*255).astype(np.uint8)

    def exact_patch(self, dst_path, rois):
        ''' Extract pathces of PolSAR data

        Args:
            dst_path (str): destination folder
            rois (list): window specifies the position of patch, should in the 
                form of [x, y, w, h], where x and y are the coordinates of the 
                lower right corner of the patch
        '''
        
        pauli = self.to_rgb()
        for ii, roi in enumerate(rois):
            dst_folder = osp.join(dst_path, str(ii))
            fu.mkdir_if_not_exist(dst_folder)

            with open(osp.join(dst_folder, 'README.txt'), 'w') as f:
                f.write(f'Original file path: {self.path}\nROI: {roi}\nin the format of (x, y, w, h), where x and y are the coordinates the lower right corner')
                raise NotImplementedError

    def _read_s2(self):
        ''' Read S2 data in envi data type

        Returns:
            s2 (ndarray): complex64 data in [channel, height, width] shape
        '''

        s11 = np.fromfile(osp.join(self.path, 's11.bin'), dtype=np.float32)
        s12 = np.fromfile(osp.join(self.path, 's12.bin'), dtype=np.float32)
        s21 = np.fromfile(osp.join(self.path, 's21.bin'), dtype=np.float32)
        s22 = np.fromfile(osp.join(self.path, 's22.bin'), dtype=np.float32)

        s11 = s11[0::2] + 1j*s11[1::2]
        s12 = s12[0::2] + 1j*s12[1::2]
        s21 = s21[0::2] + 1j*s21[1::2]
        s22 = s22[0::2] + 1j*s22[1::2]

        height = self.shape[0]
        width = self.shape[1]
        s11 = s11.reshape(height, width)
        s12 = s12.reshape(height, width)
        s21 = s21.reshape(height, width)
        s22 = s22.reshape(height, width)

        s2 = np.stack((s11, s12, s21, s22), axis=0)
        return s2
    
    def _read_c3(self):
        ''' Read C3 data in envi data type

        Returns:
            C3 (ndarray): data in [channel, height, width] shape and in 
                'save_space' storage mode
        '''

        height = self.shape[0]
        width = self.shape[1]
        
        c3 = np.empty(9, height, width)
        for ii, bin in enumerate(BIN_FIELS['C3']):
            c3[ii, ...] = np.fromfile(osp.join(self.path, bin), dtype=np.float32).reshape(height, width)

        self.storage_mode = 'save_space'
        return c3

    def _get_shape(self):
        ''' Read config file'''
        info = dict()
        with open(osp.join(self.path, 'config.txt'), 'r') as f:
            for _ in range(4):
                key = f.readline().strip()
                value = f.readline().strip()
                info[key] = value
                f.readline()
        return int(info['Nrow']), int(info['Ncol'])

    def _get_data_format(self, data_format):
        ''' Get data format '''
        if data_format is None:        
            path = self.path.lower()
            ret = None
            data_formats = ['C3', 's2', 'Hoekman', 'T3']
            for df in data_formats:
                if osp.split(path)[1] == df.lower():
                    ret = df
            if ret is None:
                raise IOError('Not a valid path or data format')
            return ret
        else:
            return data_format

"""
