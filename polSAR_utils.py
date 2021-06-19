'''
Author: Shuailin Chen
Created Date: 2021-05-19
Last Modified: 2021-06-19
	content: useful functions for polarimtric SAR data, written in early days
'''

import os
import os.path as osp
import math

import numpy as np 
from matplotlib import pyplot as plt 
import cv2
import bisect
from numpy import ndarray
import torch
from glob import glob
import xml.etree.ElementTree as et
import re
import tifffile

from mylib import file_utils as fu
from mylib import image_utils as iu
from typing import Union
from mylib import mathlib
from mylib import my_torch_tools as mt

c3_bin_files = ['C11.bin', 'C12_real.bin', 'C12_imag.bin', 'C13_real.bin', 
            'C13_imag.bin', 'C22.bin', 'C23_real.bin', 'C23_imag.bin',
            'C33.bin',]

t3_bin_files = ['T11.bin', 'T12_real.bin', 'T12_imag.bin', 'T13_real.bin', 
            'T13_imag.bin', 'T22.bin', 'T23_real.bin', 'T23_imag.bin',
            'T33.bin',]

s2_bin_files = ['s11.bin', 's12.bin', 's21.bin', 's22.bin']

hdr_elements = ['samples', 'lines', 'byte order', 'data type', 'interleave']

data_type = ['uint8', 'int16', 'int32', 'float32', 'float64', 'uint16', 'uint32', 'int64', 'uint64']

def check_c3_path(path:str)->str:
    '''check the path whether contains the c3 folder, if not, add it'''
    if path[-3:] != r'\C3' and path[-3:] != r'/C3' and (not osp.isfile(osp.join(path, 'config.txt'))):
        # print(path[-3:])
        path = os.path.join(path, 'C3')
    return path


def check_s2_path(path:str)->str:
    '''check the path whether contains the s2 folder, if not, add it'''
    if path[-3:] != r'\s2' and path[-3:] != r'/s2' and (not osp.isfile(osp.join(path, 's11.bin'))):
        path = os.path.join(path, 's2')
    return path


def read_s2_config(path:str)->dict:
    ''' read header file of S2 file'''
    path = check_s2_path(path)
    s2_info = dict()
    with open(osp.join(path, 'config.txt'), 'r') as f:
        for ii in range(4):
            key = f.readline().strip()
            value = f.readline().strip()
            s2_info[key] = value
            f.readline()
    return s2_info


def read_hdr(path:str, file='C11.bin')->dict:
    ''' Read header file of C3 file
    
    Args:
        path (str): path to the hdr file
        file (str): from whom the header file will be readed
    '''
    # path = check_c3_path(path)
    meta_info = dict()
    with open(os.path.join(path, file+'.hdr'), 'r') as hdr:
        for line in hdr:
            # print(line)
            sp = line.split('=')
            if len(sp)==2:
                ky, val = sp   
                ky = ky.strip()
                val = val.strip()
                # print(ky, ky in elements)
                if ky in hdr_elements:
                    meta_info[ky] = val
        # print(meta_info)
    return meta_info


def read_HAalpha(path):
    ''' read H/A/alpha domposed files in ENVI format 

    Args:
        path (str): folder path

    Returns:
        ndarray in (HxWx3) shape, in the order of (H, A, alpha)
    '''

    assert osp.isdir(path), 'Wrong folder path'

    meta_info = read_hdr(path, file='alpha.bin')
    H = np.fromfile(osp.join(path, 'entropy.bin'), dtype=data_type[int(meta_info['data type'])-1]).reshape(int(meta_info['lines']), int(meta_info['samples']))
    A = np.fromfile(osp.join(path, 'anisotropy.bin'), dtype=data_type[int(meta_info['data type'])-1]).reshape(int(meta_info['lines']), int(meta_info['samples']))
    alpha = np.fromfile(osp.join(path, 'alpha.bin'), dtype=data_type[int(meta_info['data type'])-1]).reshape(int(meta_info['lines']), int(meta_info['samples']))

    return np.stack((H, A, alpha), axis=0)


def read_c3(path:str, out:str='complex_vector_6', meta_info=None, count=-1, offset=0, is_print=False)->np.ndarray:
    ''' read C3 data in envi data type
    @in      -path       -path to C3 data
    @in      -out        -output format, if is 'save_space',  the last dimension of the output is the channel 
                        dimension, i.e. c11, c12_real, c12_imag, c13_real, c13_imag, c22, c23_real, c23_imag, c33
                        if is 'complex_vector_9', then the last dimension is organized as c11, c12, c13, c21, c22, 
                        c23, c31, c32, c33, so the output data will be complex numbers
                        if is 'complex_vector_6', then the last dimension is organized as c11, c12, c13, c22, c23, 
                        c33, cause the covariance matrix is conjugate symmeitric
            -meta_info  - meta info contains the .bin.hdr information
            -count      -Number of items to read. -1 means all items (i.e., the complete file).
            -offset     -The offset (in bytes) from the files current position. 
    @out     -C3 data in the specified format, 3-D matrix shape of [channel x height x width]
    '''

    path = check_c3_path(path)

    if is_print:
        print('reading from ', path)

    if meta_info is None:
        meta_info = read_hdr(path)

    # read binary files
    c11 = np.fromfile(os.path.join(path, 'C11.bin'), dtype=data_type[int(meta_info['data type'])-1], count=count, offset=offset)
    c11 = c11.reshape(int(meta_info['lines']), int(meta_info['samples']))

    # print(os.path.join(path, 'C12_real.bin'))
    c12_real = np.fromfile(os.path.join(path, 'C12_real.bin'), dtype=data_type[int(meta_info['data type'])-1], count=count, offset=offset)
    c12_real = c12_real.reshape(int(meta_info['lines']), int(meta_info['samples']))
    
    c12_img = np.fromfile(os.path.join(path, 'C12_imag.bin'), dtype=data_type[int(meta_info['data type'])-1], count=count, offset=offset)
    c12_img = c12_img.reshape(int(meta_info['lines']), int(meta_info['samples']))
    
    c13_real = np.fromfile(os.path.join(path, 'C13_real.bin'), dtype=data_type[int(meta_info['data type'])-1], count=count, offset=offset)
    c13_real = c13_real.reshape(int(meta_info['lines']), int(meta_info['samples']))
    
    c13_img = np.fromfile(os.path.join(path, 'C13_imag.bin'), dtype=data_type[int(meta_info['data type'])-1], count=count, offset=offset)
    c13_img = c13_img.reshape(int(meta_info['lines']), int(meta_info['samples']))

    c22 = np.fromfile(os.path.join(path, 'C22.bin'), dtype=data_type[int(meta_info['data type'])-1], count=count, offset=offset)
    c22 = c22.reshape(int(meta_info['lines']), int(meta_info['samples']))
    
    c23_real = np.fromfile(os.path.join(path, 'C23_real.bin'), dtype=data_type[int(meta_info['data type'])-1], count=count, offset=offset)
    c23_real = c23_real.reshape(int(meta_info['lines']), int(meta_info['samples']))
    
    c23_img = np.fromfile(os.path.join(path, 'C23_imag.bin'), dtype=data_type[int(meta_info['data type'])-1], count=count, offset=offset)
    c23_img = c23_img.reshape(int(meta_info['lines']), int(meta_info['samples']))
    
    c33 = np.fromfile(os.path.join(path, 'C33.bin'), dtype=data_type[int(meta_info['data type'])-1], count=count, offset=offset)
    c33 = c33.reshape(int(meta_info['lines']), int(meta_info['samples']))

    # constructe to the specified data format
    if out == 'save_space':
        c3 = np.stack((c11, c12_real, c12_img, c13_real, c13_img, c22, c23_real, c23_img, c33), axis=0)
    elif out == 'complex_vector_9':
        c12 = c12_real + 1j*c12_img
        c13 = c13_real + 1j*c13_img
        c23 = c23_real + 1j*c23_img
        c3 = np.stack((c11, c12, c13, c12.conj(), c22, c23, c13.conj(), c23.conj(), c33), axis=0)
    elif out == 'complex_vector_6':
        c12 = c12_real + 1j*c12_img
        c13 = c13_real + 1j*c13_img
        c23 = c23_real + 1j*c23_img
        c3 = np.stack((c11, c12, c13,  c22, c23,  c33), axis=0)
    else:
        raise LookupError('wrong output format')
    return c3


def read_s2(path:str, meta_info=None, count=-1, offset=0, is_print=None)->np.ndarray:
    ''' read S2 data in envi data type
    @in      -path       -path to S2 data
    @in     -meta_info  - meta info contains the .bin.hdr information
            -count      -Number of items to read. -1 means all items (i.e., the complete file).
            -offset     -The offset (in bytes) from the files current position. 
    @out     -S2 data in the torch.complex64 format, 3-D matrix shape of [channel x height x width]
    '''
    path = check_s2_path(path)
    if is_print:
        print('reading from ', path)

    if meta_info is None:
        meta_info = read_s2_config(path)

    s11 = np.fromfile(osp.join(path, 's11.bin'), dtype=np.float32, count=count, offset=offset)
    s12 = np.fromfile(osp.join(path, 's12.bin'), dtype=np.float32, count=count, offset=offset)
    s21 = np.fromfile(osp.join(path, 's21.bin'), dtype=np.float32, count=count, offset=offset)
    s22 = np.fromfile(osp.join(path, 's22.bin'), dtype=np.float32, count=count, offset=offset)

    s11 = s11[0::2] + 1j*s11[1::2]
    s12 = s12[0::2] + 1j*s12[1::2]
    s21 = s21[0::2] + 1j*s21[1::2]
    s22 = s22[0::2] + 1j*s22[1::2]

    height = int(meta_info['Nrow'])
    weight = int(meta_info['Ncol'])
    s11 = s11.reshape(height, weight)
    s12 = s12.reshape(height, weight)
    s21 = s21.reshape(height, weight)
    s22 = s22.reshape(height, weight)

    s2 = np.stack((s11, s12, s21, s22), axis=0)
    return s2


def read_c3_GF3_L2(path, is_print=False):
    ''' Read 4 channel (HH, HV, VH, VV) data from Gaofen-3 L2 data

    Args:
        path (str): folder to the product file
        is_print (bool): if to print infos

    Returns:
        img (ndarray): four channels data in [channel, height, weight] shape, logarithmized
	'''
    
    if is_print:
        print(f'Reading GF3 L2 data from {path}')

    # seek for tiff files
    tifs = glob(osp.join(path, '*.tiff'))
    tifs.sort()

    # read qualify value and calibration constant (K_dB)
    meta_xml_path = glob(osp.join(path, '*.meta.xml'))
    root = et.parse(osp.join(meta_xml_path[0])).getroot()

    calibrate_const = []
    for item in root.iter('CalibrationConst'):
        for pol in item:
            calibrate_const.append(pol.text)
    calibrate_const = np.array(calibrate_const).reshape(-1, 1, 1).astype(np.float32)
    
    qualify_value = []
    for item in root.iter('QualifyValue'):
        for pol in item:
            qualify_value.append(pol.text)
    qualify_value = np.array(qualify_value).reshape(-1, 1, 1).astype(np.float32)

    # read tiff
    img = [tifffile.imread(tif) for tif in tifs]
    img = np.stack(img, axis=0)
    if is_print:
        print(f'image shape: {img.shape}\n')    

    # calibrate
    img = img.astype(np.float32) 
    img[img<mathlib.eps] = mathlib.eps
    img = 10*np.log10(img**2 * (qualify_value/65535)**2) - calibrate_const

    return img
    

def read_s2_GF3_L1A(path, file_ext='tiff', is_print=False):
    ''' Read 4 channel (HH, HV, VH, VV) data from Gaofen-3 L1A data, discarding calibration const!!!

    Args:
        path (str): folder to the product file
        file_ext (Str): file extern. Default: tiff
        is_print (bool): if to print infos

    Returns:
        img (ndarray): four channels data in [channel, height, weight] shape
	'''
    
    if is_print:
        print(f'Reading GF3 L1A data from {path}')

    # seek for tiff files
    tifs = glob(osp.join(path, '*.'+file_ext))
    tifs.sort()

    # read qualify value and calibration constant (K_dB)
    meta_xml_path = glob(osp.join(path, '*.meta.xml'))
    root = et.parse(osp.join(meta_xml_path[0])).getroot()
    
    qualify_value = []
    for item in root.iter('QualifyValue'):
        for pol in item:
            qualify_value.append(pol.text)
    qualify_value = np.array(qualify_value).reshape(-1, 1, 1).astype(np.float32)
    qualify_value = np.repeat(qualify_value, 2)
    qualify_value = qualify_value[:, np.newaxis, np.newaxis]

    # read tiff
    img = [tifffile.imread(tif) for tif in tifs]
    img = np.concatenate(img, axis=0)
    if is_print:
        print(f'image shape: {img.shape}\n')    

    # calibrate
    img = img.astype(np.float32) 
    # img[img<mathlib.eps] = mathlib.eps
    img = img * qualify_value / 32767

    cimg = np.empty(shape=(4, *img.shape[1:]), dtype=np.complex64)
    cimg.real = img[::2, ...]
    cimg.imag = img[1::2, ...]
    return cimg
    

def s22c3(path=None, s2=None):
    ''' Convert s2 data to C3 data

    Args:
        path (str): path of s2 data
        s2 (ndarray): s2 data, should not be used if "path" is specified

    Returns:
        converted C3 data in 'complex_vector_9' data format
    '''
    
    if path is not None:
        s2 = read_s2(path)
    
    # lexicographic basis in reciprocal condition
    _, h, w = s2.shape
    kl = s2.copy()
    kl[1, ...] = (kl[1, ...]+kl[2, ...]) / np.sqrt(2)
    kl = kl[[0, 1, 3], ...]

    # s2 convert to C3
    C3 = np.einsum('ij..., jk...->ik...', kl[:, np.newaxis, ...], kl[np.newaxis, ...].conj())

    return C3.reshape(9, h ,w)

    
def c32t3(path: str=None, c3: ndarray=None) -> ndarray :
    ''' change C3 data to T3 data 
    @in     -path       -path of c3 data
            -C3         -C3 data, should not be used if "path" is specified
    @ret    -T3         -T3 data
    '''
    if path:
        c3_ = read_c3(path, out='save_space')
    else:
        c3_  = as_format(c3, 'save_space')
    
    t3 = np.zeros_like(c3_)
    t3[0, ...] = (c3_[0, ...] + 2*c3_[3, ...] + c3_[8, ...]) / 2
    t3[1, ...] = (c3_[0, ...] - c3_[8, ...]) / 2
    t3[2, ...] = -c3_[4, ...]
    t3[3, ...] = (c3_[1, ...] + c3_[6, ...]) / np.sqrt(2)
    t3[4, ...] = (c3_[2, ...] - c3_[7, ...]) / np.sqrt(2)
    t3[5, ...] = (c3_[0, ...] - 2*c3_[3, ...] + c3_[8, ...]) / 2
    t3[6, ...] = (c3_[1, ...] - c3_[6, ...]) / np.sqrt(2)
    t3[7, ...] = (c3_[2, ...] + c3_[7, ...]) / np.sqrt(2)
    t3[8, ...] = c3_[5, ...]  

    return t3


def write_config_hdr(path:str, config:Union[dict, list, tuple], config_type='hdr', data_type='c3')->None:
    """
    write config.txt file and Cxx.hdr file
    @in     -path           -data path
            -config         -config information require for .bin.hdr file, in a dict format
            -config_type    -config type, 'hdr' or 'cfg'
    """
    if config_type=='hdr':
        if isinstance(config, dict):
            lines = config['lines']
            samples = config['samples']
            datatype = config["data type"]
            interleave = config["interleave"]
            byteorder = config["byte order"]
        elif isinstance(config, (list, tuple)):
            lines = str(config[0])
            samples = str(config[1])
            datatype = '4'
            interleave = 'bsq'
            byteorder = '0'
        
        if data_type=='c3':
            bin_files = c3_bin_files
        elif data_type=='t3':
            bin_files = t3_bin_files
        elif data_type=='s2':
            bin_files = s2_bin_files
        else:
            raise ValueError('unrecognized data type')

        for bin in bin_files:
            file_hdr = osp.join(path, bin + '.hdr')
            with open(file_hdr, 'w') as hdr:
                hdr.write('ENVI\ndescription = {File Imported into ENVI.}\n')
                hdr.write(f'samples = {samples}\n')
                hdr.write(f'lines = {lines}\n')
                hdr.write('bands = 1\n')
                hdr.write('header offset = 0\nfile type = ENVI Standard\n')
                hdr.write(f'data type = {datatype}\n')
                hdr.write(f'interleave = {interleave}\n')
                hdr.write('sensor type = Unknown\n')
                hdr.write(f'byte order = {byteorder}\n')
                hdr.write(f'band names = {{{bin}}}\n')
    elif config_type=='cfg':
        lines = config['Nrow']
        samples = config['Ncol']
    else:
        raise NotImplementedError

    with open(osp.join(path, 'config.txt'), 'w') as cfg:
        cfg.write('Nrow\n')     # nrow = lines, ncol = smaples
        cfg.write(lines)
        cfg.write('\n---------\nNcol\n')
        cfg.write(samples)
        cfg.write('\n---------\n')
        cfg.write('PolarCase\nmonostatic\n---------\nPolarType\nfull')
    


def write_c3(path:str, data:ndarray, config:dict=None, config_type='hdr', is_print=False):    
    ''' 
    write c3 data 
    @in     -path       -data path
            -data       -the polSAR data
            -config     -config information require for .bin.hdr file, in a dict format
            -is_print   -whether to print the debug info
    '''
    
    # check input
    if is_print:
        print('writing ', path)
    fu.mkdir_if_not_exist(path)

    # write config.txt and *.bin.hdr file
    if config is None:
        config = data.shape[1:]
    write_config_hdr(path, config, config_type)

    # write binary files
    if (isinstance(config, dict) and config['data type'] == '4') or isinstance(config, (list, tuple)):
        data = as_format(data, out='save_space')
        for idx, bin in enumerate(c3_bin_files):
            fullpath = osp.join(path, bin)
            file = data[idx, :, :]
            file.tofile(fullpath)
    else:
        raise NotImplementedError('data type is not float32')


def write_t3(path:str, data:ndarray, config:dict=None, is_print=False):    
    ''' 
    write t3 data 
    @in     -path       -data path
            -data       -the polSAR data
            -config     -config information require for .bin.hdr file, in a dict format
            -is_print   -whether to print the debug info
    '''
    
    # check input
    if is_print:
        print('writing ', path)

    # write config.txt and *.bin.hdr file
    if config is None:
        config = data.shape[1:]
    write_config_hdr(path, config, data_type='t3')


    # write binary files
    if (isinstance(config, dict) and config['data type'] == '4') or isinstance(config, (list, tuple)):
        data = as_format(data, out='save_space')
        for idx, bin in enumerate(t3_bin_files):
            fullpath = osp.join(path, bin)
            file = data[idx, :, :]
            file.tofile(fullpath)
    else:
        raise NotImplementedError('data type is not float32')


def write_s2(path:str, data:ndarray, config:dict=None, is_print=False, config_type='hdr'):    
    ''' 
    write s2 data as envi data format
    @in     -path       -data path
            -data       -the polSAR data
            -config     -config information require for .bin.hdr file, in a dict format
            -is_print   -whether to print the debug info
    '''
    # check input
    if is_print:
        print('writign ', path)

    # write config.txt
    if config is None:
        config = data.shape[1:]
    write_config_hdr(path, config, config_type=config_type, data_type='s2')

    # write binary files
    if data.dtype==np.complex64:
        for idx, bin in enumerate(s2_bin_files):
            full_path = osp.join(path, bin)
            file = data[idx, :, :]
            file.tofile(full_path)
    else:
        raise NotImplementedError('data type should be np.complex64')


def read_bmp(path:str, is_print=None)->np.ndarray:
    '''@brief   -read bmp image file
    @in      -path          -path to C3 data
    @out     -bmp image file, uint8 dtype
    '''
    # check input
    path = check_c3_path(path)

    if is_print:
        print('reading from ', path)

    return cv2.imread(osp.join(path, 'PauliRGB.bmp'))


def as_format(data:np.ndarray, out:str='save_space')->np.ndarray:
    '''@brief   -change the data organization format
   @in      -data   -the data need to be transformed
   @in      -out    -output data format, either 'save_space' or 'complex_vector_6' or 'complex_vector_9', 
                    can't be the same with the input data format
   @out     -transformed data
    '''
    # decide the input data format
    ch = data.shape[0]
    # complex vector
    if np.iscomplexobj(data):
        # complex_vector_6
        if ch==6:   
            if out == 'save_space':
                c11 = data[0, :, :].real
                c12_real = data[1, :, :].real
                c12_img = data[1, :, :].imag
                c13_real = data[2, :, :].real
                c13_img = data[2, :, :].imag
                c22 = data[3, :, :].real
                c23_real = data[4, :, :].real
                c23_img = data[4, :, :].imag
                c33 = data[5, :, :].real
                c3 = np.stack((c11, c12_real, c12_img, c13_real, c13_img, c22, c23_real, c23_img, c33), axis=0)
            elif out == 'complex_vector_9':
                c11 = data[0, :, :]
                c12 = data[1, :, :]
                c13 = data[2, :, :]
                c22 = data[3, :, :]
                c23 = data[4, :, :]
                c33 = data[5, :, :]
                c3 = np.stack((c11, c12, c13, c12.conj(), c22, c23, c13.conj(), c23.conj(), c33), axis=0)
            elif out == 'complex_vector_6':
                c3 = data
            else:
                raise LookupError('wrong out format')

        # complex_vector_9
        elif ch==9: 
            if out == 'save_space':
                c11 = data[0, :, :].real
                c12_real = data[1, :, :].real
                c12_img = data[1, :, :].imag
                c13_real = data[2, :, :].real
                c13_img = data[2, :, :].imag
                c22 = data[4, :, :].real
                c23_real = data[5, :, :].real
                c23_img = data[5, :, :].imag
                c33 = data[8, :, :].real
                c3 = np.stack((c11, c12_real, c12_img, c13_real, c13_img, c22, c23_real, c23_img, c33), axis=0)
            elif out == 'complex_vector_6':
                c11 = data[0, :, :]
                c12 = data[1, :, :]
                c13 = data[2, :, :]
                c22 = data[4, :, :]
                c23 = data[5, :, :]
                c33 = data[8, :, :]
                c3 = np.stack((c11, c12, c13, c22, c23, c33), axis=0)
            elif out == 'complex_vector_9':
                c3 = data
            else:
                raise LookupError('wrong out format')
        else:
            raise LookupError('wrong shape of input data')
        
    # save_space
    elif ch==9 :
        if out == 'complex_vector_9':
            c11 = data[0, :, :]
            c12 = data[1, :, :] + 1j*data[2, :, :]
            c13 = data[3, :, :] + 1j*data[4, :, :]
            c22 = data[5, :, :]
            c23 = data[6, :, :] + 1j*data[7, :, :]
            c33 = data[8, :, :]
            c3 = np.stack((c11, c12, c13, c12.conj(), c22, c23, c13.conj(), c23.conj(), c33), axis=0)
        elif out == 'complex_vector_6':
            c11 = data[0, :, :]
            c12 = data[1, :, :] + 1j*data[2, :, :]
            c13 = data[3, :, :] + 1j*data[4, :, :]
            c22 = data[5, :, :]
            c23 = data[6, :, :] + 1j*data[7, :, :]
            c33 = data[8, :, :]
            c3 = np.stack((c11, c12, c13,  c22, c23,  c33), axis=0)
        elif out == 'save_space':
            c3 = data
        else:
            raise LookupError('wrong out format')
    else:
        raise LookupError('wrong format of input data')
    return c3


def rgb_by_c3(data:np.ndarray, type:str='pauli', is_print=False)->np.ndarray:
    ''' Create the pseudo RGB image with covariance matrix

    Args:
        data (ndarray): input polSAR data
        type (str): 'pauli' or 'sinclair'. Default: 'pauli'
        is_print (bool): if to print debug infos. Default: False

    Returns:
        RGB data in [0, 1]
    '''
    type = type.lower()
    data = as_format(data, out='complex_vector_6')

    # compute orginal RGB components
    if type == 'pauli':
        # print('test')
        R = 0.5*(data[0, :, :]+data[5, :, :])-2*data[2, :, :]
        G = data[3, :, :]
        B = 0.5*(data[0, :, :]+data[5, :, :])+2*data[2, :, :]
    elif type == 'sinclair':
        R = data[5, :, :]
        G = data[3, :, :]
        B = data[0, :, :]

    # print(R, '\n')
    # abs
    R = np.abs(R)
    G = np.abs(G)
    B = np.abs(B)

    # clip
    R[R<mathlib.eps] = mathlib.eps
    G[G<mathlib.eps] = mathlib.eps
    B[B<mathlib.eps] = mathlib.eps

    # logarithm 
    R = 10*np.log10(R)
    G = 10*np.log10(G)
    B = 10*np.log10(B)

    # _TMP_PATH = r'/home/csl/code/PolSAR_N2N/tmp'
    # fig = iu.plot_surface(R)
    # plt.savefig(osp.join(_TMP_PATH, 'R.jpg'))
    # plt.show()
    # plt.clf()
    # fig = iu.plot_surface(G)
    # plt.savefig(osp.join(_TMP_PATH, 'R.jpg'))
    # plt.show()
    # plt.clf()
    # fig = iu.plot_surface(B)
    # plt.savefig(osp.join(_TMP_PATH, 'R.jpg'))
    # plt.show()

    # normalize
    R = mathlib.min_max_contrast_median_map(R, is_print=is_print)
    G = mathlib.min_max_contrast_median_map(G, is_print=is_print)
    B = mathlib.min_max_contrast_median_map(B, is_print=is_print)

    # print(R.shape, G.shape, B.shape)
    return np.stack((R, G, B), axis=2)


def rgb_by_t3(data:np.ndarray, type:str='pauli')->np.ndarray:
    ''' @brief   -create the pseudo RGB image with covariance matrix
    @in      -data  -input polSAR data
    @in      -type  -'pauli' or 'sinclair'
    @out     -RGB data in [0, 1]
    '''
    type = type.lower()

    data = as_format(data, out='complex_vector_6')

    # compute orginal RGB components
    if type == 'pauli':
        # print('test')
        R = data[3, :, :]
        G = data[5, :, :]
        B = data[0, :, :]

    # print(R, '\n')
    # abs
    R = np.abs(R)
    G = np.abs(G)
    B = np.abs(B)

    # clip
    R[R<mathlib.eps] = mathlib.eps
    G[G<mathlib.eps] = mathlib.eps
    B[B<mathlib.eps] = mathlib.eps

    # print(R, '\n')
    # logarithm 
    R = 10*np.log10(R)
    G = 10*np.log10(G)
    B = 10*np.log10(B)
    
    # normalize
    # R = min_max_contrast_median_map(R[R!=10*np.log10(mathlib.eps)])
    # G = min_max_contrast_median_map(G[G!=10*np.log10(mathlib.eps)])
    # B = min_max_contrast_median_map(B[B!=10*np.log10(mathlib.eps)])
    R = mathlib.min_max_contrast_median_map(R)
    G = mathlib.min_max_contrast_median_map(G)
    B = mathlib.min_max_contrast_median_map(B)

    # print(R.shape, G.shape, B.shape)
    return np.stack((R, G, B), axis=2)


def rgb_by_s2(data:np.ndarray, type:str='pauli', if_log=True, if_mask=False)->np.ndarray:
    ''' Create the pseudo RGB image with s2 matrix

    Args:
        data (ndarray): input polSAR data, in shape of [channel, height, 
            weight]
        type (str): 'pauli' or 'sinclair'. Default: pauli
        if_log (bool): if do logarithm to data. Default: True
        if_mask (bool): if to set mask to the invalid data, preventing it 
            from computing the upper and lower bound
    Returns:
        RGB data in [0, 255]
    '''

    type = type.lower()

    s11 = data[0, :, :]
    s12 = data[1, :, :]
    s21 = data[2, :, :]
    s22 = data[3, :, :]

    if type == 'pauli':
        assert not np.all(np.isreal(data))
        R = 0.5*np.conj(s11-s22)*(s11-s22)
        G = 0.5*np.conj(s12+s21)*(s12+s21)
        B = 0.5*np.conj(s11+s22)*(s11+s22)

    elif type == 'sinclair':
        R = s22
        G = (s12+s21) / 2
        B = s11

    # abs if complex data
    if not np.all(np.isreal(data)):
        R = np.abs(R)
        G = np.abs(G)
        B = np.abs(B)

    # logarithm transform, and normalize
    if if_log:
        R[R<mathlib.eps] = mathlib.eps
        G[G<mathlib.eps] = mathlib.eps
        B[B<mathlib.eps] = mathlib.eps

        R = 10*np.log10(R)
        G = 10*np.log10(G)
        B = 10*np.log10(B)
    
    # mask the valid pixels
    R_mask = None
    G_mask = None
    B_mask = None
    if if_mask:
        R_mask = R > -150
        G_mask = G > -150
        B_mask = B > -150

    # min map map
    R = mathlib.min_max_contrast_median_map(R, mask=R_mask)
    G = mathlib.min_max_contrast_median_map(G, mask=G_mask)
    B = mathlib.min_max_contrast_median_map(B, mask=B_mask)

    return (np.stack((R, G, B), axis=2)*255).astype(np.uint8)


def write_hoekman_image(data, dst_path, is_print=False):
    ''' write each channel of hoekman data to a separated grayscale image

    Args:
        data (ndarray): hoekman coefficient, in shape of 
            [channel, height, width]
        dst_path (str): path to save images
        if_print (bool): if the print debug infos
    '''
    for ii in range(9):
        gray_img = mathlib.min_max_contrast_median_map(10*np.log10(data[ii, :, :]), is_print=is_print)
        iu.save_image_by_cv2(gray_img, dst_path=osp.join(dst_path, f'{ii}.png'))


def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    ''' @brief  -the same as imadjust() function in matlab 
        @note   -there seems to be some bugs
    '''
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    dst = src.copy()
    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[src[r,c]] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            vs = max(src[r,c] - vin[0], 0)
            vd = min(int(vs * scale + 0.5) + vout[0], vout[1])
            dst[r,c] = vd
    return dst


def exact_patch_C3(src_path, roi, dst_path=None):
    ''' Extract pathces of C3 data

    Args:
        src_path (str): source folder 
        dst_path (str): destination folder
        roi (list): window specifies the position of patch, should in the 
            form of [x, y, w, h], where x and y are the coordinates of the 
            lower right corner of the patch
    '''
    if dst_path is None:
        dst_path = src_path
    print(f'extract c3 data from {src_path},\nto {dst_path},\nrois: {roi}')
    
    c3 = read_c3(src_path)
    pauli = rgb_by_c3(c3)
    
    fu.mkdir_if_not_exist(dst_path)

    with open(osp.join(dst_path, 'README.txt'), 'w') as f:
        f.write(f'Original file path: {src_path}\nROI: {roi}\nin the format of (x, y, w, h), where x and y are the coordinates the lower right corner')

    xs = roi[0] - roi[2]+1
    ys = roi[1] - roi[3]+1
    xe = roi[0] + 1
    ye = roi[1] + 1
    c3 = c3[:, ys:ye, xs:xe]
    # write_c3(dst_path, c3, {'Nrow': roi[3], 'Ncol': roi[2]}, 'cfg')
    write_c3(dst_path, c3)
    
    pauli_roi = pauli[ys:ye, xs:xe, :]
    cv2.imwrite(osp.join(dst_path, 'pauliRGB.bmp'), cv2.cvtColor((pauli_roi*255).astype(np.uint8), cv2.COLOR_BGR2RGB))


def exact_patch_s2(src_path, roi, dst_path=None):
    ''' Extract pathces of s2 data

    Args:
        src_path (str): source folder 
        dst_path (str): destination folder
        roi (list): window specifies the position of patch, should in the 
            form of [x, y, w, h], where x and y are the coordinates of the 
            lower right corner of the patch
    '''
    if dst_path is None:
        dst_path = src_path
    print(f'extract s2 data from {src_path},\nto {dst_path},\nrois: {roi}')
    
    s2 = read_s2(src_path)
    pauli = rgb_by_s2(s2)
    
    fu.mkdir_if_not_exist(dst_path)

    with open(osp.join(dst_path, 'README.txt'), 'w') as f:
        f.write(f'Original file path: {src_path}\nROI: {roi}\nin the format of (x, y, w, h), where x and y are the coordinates the lower right corner')

    xs = roi[0] - roi[2]+1
    ys = roi[1] - roi[3]+1
    xe = roi[0] + 1
    ye = roi[1] + 1
    s2 = s2[:, ys:ye, xs:xe]
    write_s2(dst_path, s2)
    
    pauli_roi = pauli[ys:ye, xs:xe, :]
    # cv2.imwrite(osp.join(dst_path, 'pauliRGBwhole.png'), (pauli*255).astype(np.uint8))
    cv2.imwrite(osp.join(dst_path, 'pauliRGB.png'), pauli_roi)


def split_patch(path, patch_size=[512, 512], transpose=False)->None:
    ''' 
    split the who image into several patches 
    @in     -path           -path to C3 data
            -patch_size     -size of a patch, in [height, width] format
    '''
    print('working dir : ', path)
    whole_config = read_hdr(path)
    print(whole_config)
    whole_data = read_c3(path=path, out='save_space', meta_info=whole_config)
    whole_img = read_bmp(path)
    if whole_img is None:           #没有 pauliRGB 的话就生成
        whole_img = rgb_by_c3(whole_data)*255
        whole_img = cv2.cvtColor(whole_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(osp.join(path, 'PauliRGB.bmp'), whole_img)
    if transpose:
        whole_data = whole_data.transpose((0, 2, 1))
        whole_img = rgb_by_c3(whole_data)*255
        whole_img = cv2.cvtColor(whole_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(osp.join(path, 'PauliRGB.bmp'), whole_img)
        tmp = whole_config['lines']
        whole_config['lines'] = whole_config['samples']
        whole_config['samples'] = tmp
        write_c3(path, whole_data, whole_config, is_print=True)
        cv2.imwrite(osp.join(path, 'PauliRGB.bmp'), whole_img)
    whole_het, whole_wes = whole_img.shape[:2]
    idx = 0
    start_x = 0
    start_y = 0
    p_het, p_wes = patch_size
    whole_config['samples'] = '512'
    whole_config['lines'] = '512'
    while start_x<whole_wes and start_y<whole_het:
        print(f'    spliting the {idx}-th patch')

        # write bin file
        p_data = whole_data[:, start_y:start_y+p_het, start_x:start_x+p_wes]
        p_folder = osp.join(path, str(idx))
        fu.mkdir_if_not_exist(p_folder)
        write_c3(p_folder, p_data, whole_config, is_print=True)

        # write pauliRGB, which is cutted from big picture, not re-generated 
        p_img = whole_img[start_y:start_y+p_het, start_x:start_x+p_wes, :]
        cv2.imwrite(osp.join(p_folder, 'PauliRGB.bmp'), p_img)

        # increase patch index
        idx += 1
        start_x += p_wes
        if start_x >= whole_wes:      # next row
            start_x = 0
            start_y += p_het
            if start_y>=whole_het:          # finish
                print('totle split', idx, 'patches done')
                return
            elif start_y+p_het > whole_het: # suplement
                start_y = whole_het - p_het
        elif start_x+p_wes > whole_wes: 
            start_x = whole_wes - p_wes      


def split_patch_s2(path, patch_size=(512, 512), transpose=False)->None:
    ''' 
    split the who image into several patches 
    @in     -path           -path to s2 data
            -patch_size     -size of a patch, in [height, width] format
            -tranpose       -whether to transpose the spatial axes of data
    '''
    path = check_s2_path(path)
    print('spliting the dir: ', path)

    whole_cfg = read_s2_config(path)
    print('config: ', whole_cfg)
    whole_data = read_s2(path, meta_info=whole_cfg)
    whole_img = read_bmp(path)

    if whole_img is None:
        whole_img = rgb_by_s2(whole_data)*255
        whole_img = cv2.cvtColor(whole_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(osp.join(path, 'PauliRGB.bmp'), whole_img)

    if transpose:
        whole_data = whole_data.transpose((0,2,1))
        whole_img = rgb_by_s2(whole_data)*255
        whole_img = cv2.cvtColor(whole_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(osp.join(path, 'PauliRGB.bmp'), whole_img)
        whole_cfg['Nrow'], whole_cfg['Ncol'] = whole_cfg['Ncol'], whole_cfg['Nrow']
        write_s2(path, whole_data, whole_cfg)

    whole_het, whole_wes = int(whole_cfg['Nrow']), int(whole_cfg['Ncol'])
    idx = 0
    start_x = 0
    start_y = 0
    p_het, p_wes = patch_size
    whole_cfg['Nrow'] = '512'
    whole_cfg['Ncol'] = '512'
    while start_x<whole_wes and start_y<whole_het:
        print(f'    spliting the {idx}-th patch')

        # write bin file
        p_data = whole_data[:, start_y:start_y+p_het, start_x:start_x+p_wes]
        p_folder = osp.join(path, str(idx))
        fu.mkdir_if_not_exist(p_folder)
        write_s2(p_folder, p_data, whole_cfg, is_print=True)

        # write pauliRGB, which is cutted from big picture, not re-generated 
        p_img = whole_img[start_y:start_y+p_het, start_x:start_x+p_wes, :]
        cv2.imwrite(osp.join(p_folder, 'PauliRGB.bmp'), p_img)

        # increase patch index
        idx += 1
        start_x += p_wes
        if start_x >= whole_wes:      # next row
            start_x = 0
            start_y += p_het
            if start_y>=whole_het:          # finish
                print('totle split', idx, 'patches done')
                return
            elif start_y+p_het > whole_het: # suplement
                start_y = whole_het - p_het
        elif start_x+p_wes > whole_wes: 
            start_x = whole_wes - p_wes   


def split_patch_HAalpha(path, patch_size=[512, 512], transpose=False)->None:
    ''' Split the who image into several patches 
    Args:
        path (str): path to C3 data
        patch_size (list or tuple): size of a patch, in [height, width] format
        transpose (bool): whether to transpose the images
    '''

    if osp.isdir(path):
        dst_path = osp.join(path, 'HAalpha')
        path = osp.join(path, 'HAalpha.npy')
    elif osp.isfile(path):
        dst_path = osp.join(osp.dirname(path), 'HAalpha')
    else:
        raise IOError('Not a valid path')

    print('working on: ', path)
    whole_data = np.load(path)

    if transpose:
        whole_data = whole_data.transpose((0, 2, 1))
    
    fu.mkdir_if_not_exist(dst_path)
    np.save(osp.join(dst_path, 'unnormed.npy'), whole_data)

    whole_het, whole_wes = whole_data.shape[1:]
    idx = 0
    start_x = 0
    start_y = 0
    p_het, p_wes = patch_size
    while start_x<whole_wes and start_y<whole_het:
        print(f'    spliting the {idx}-th patch')

        # write bin file
        p_data = whole_data[:, start_y:start_y+p_het, start_x:start_x+p_wes]
        p_folder = osp.join(dst_path, str(idx))
        fu.mkdir_if_not_exist(p_folder)
        np.save(osp.join(p_folder, 'unnormed.npy'), p_data)

        # increase patch index
        idx += 1
        start_x += p_wes
        if start_x >= whole_wes:      # next row
            start_x = 0
            start_y += p_het
            if start_y>=whole_het:          # finish 
                os.remove(path)         # delete original file
                print('totle split', idx, 'patches done')
                return
            elif start_y+p_het > whole_het: # suplement
                start_y = whole_het - p_het
        elif start_x+p_wes > whole_wes: 
            start_x = whole_wes - p_wes    


def Hokeman_decomposition(data:ndarray, if_scale=False)->ndarray:
    ''' Calculate the Hokeman decomposition, which transforms the C3 matrix into 9 independent SAR intensities 

    Args:
        data (ndarray): data to be transformed, in [channel, height, width]
            format
        if_scale (bool): if to scale the data by 4*Pi. Default: False

    Returns:
        H (ndarray): transformed Hoekman coefficient, in [channel, height,
            width] format
    '''

    vb = np.array([[   1,   0,   0,    0,    0,   0,    0,   0,    0],
                [   0,   1,   0,    0,    0,   0,    0,   0,    0],
                [ 1/4, 1/4,   1,  1/2,    0,   1,    0,   1,    0],
                [ 1/4, 1/4,   1,  1/2,    0,  -1,    0,  -1,    0],
                [ 1/4, 1/4,   1, -1/2,    0,   0,   -1,   0,   -1],
                [ 1/4, 1/4,   1, -1/2,    0,   0,    1,   0,    1],
                [ 1/2,   0, 1/2,    0,    0,   1,    0,   0,    0],
                [ 1/2,   0, 1/2,    0,    0,   0,   -1,   0,    0],
                [ 1/4, 1/4, 1/2,    0, -1/2, 1/2, -1/2, 1/2, -1/2]], dtype=np.float32)
    if if_scale:
        vb *= 4 * np.pi

    data = as_format(data, 'save_space')
    data = data[[0,8,5,3,4,1,2,6,7], ...]
    _, m, n = data.shape
    H = (vb @ data.reshape(9, -1)).reshape(9, m, n)
    # H *= 4*np.pi
    # H[H<mathlib.eps] = mathlib.eps
    return H


def inverse_Hokeman_decomposition(data:ndarray, if_scale=False)->ndarray:
    ''' Calculate inverse Hokeman decomposition, which transforms the 9 independent SAR intensities into C3 matrix 

    Args:
        data (ndarray): Hoekman coefficient to be transformed, in [channel,
            height, width] format
        if_scale (bool): if to scale the data by 1/(4*Pi). Default: False

    Returns:
        C3 (ndarray): transformed C3 data, in 'save_space' data format
    '''

    b = np.array([[    1,    0,    0,    0,    0,    0,  0,  0,  0],
                    [    0,    1,    0,    0,    0,    0,  0,  0,  0],
                    [ -1/4, -1/4,  1/4,  1/4,  1/4,  1/4,  0,  0,  0],
                    [    0,    0,  1/2,  1/2, -1/2, -1/2,  0,  0,  0],
                    [  1/4,  1/4,  3/4, -1/4,  3/4, -1/4,  0,  0, -2],
                    [ -3/8,  1/8, -1/8, -1/8, -1/8, -1/8,  1,  0,  0],
                    [  3/8, -1/8,  1/8,  1/8,  1/8,  1/8,  0, -1,  0],
                    [  3/8, -1/8,  5/8, -3/8,  1/8,  1/8, -1,  0,  0],
                    [ -3/8,  1/8, -1/8, -1/8, -5/8,  3/8,  0,  1,  0]], dtype=np.float32)
    if if_scale:
        b /= 4 * np.pi

    _, m, n = data.shape
    C3 = b @ data.reshape(9, -1)
    C3 = C3.reshape(9, m, n)
    # C3[C3<mathlib.eps] = mathlib.eps
    # C3 = 4*np.pi
    C3 = C3[[0, 5, 6, 3, 4, 2, 7, 8, 1], ...]

    return C3


def my_cholesky(M, dtype='torch'):
    """
    Compute the cholesky decomposition of a number of SPD matrix M.
    @in     -M      -assume in [height, width, channel] or [height, width] shape
    @in     -dtype  -'torch' or 'numpy'
    @ret    -L      -lower trianglar
    note: no checking is perfomered to verify whether M is hermitian and semi positive definite or not.
    """

    A = np.copy(M)
    L = np.zeros_like(A)
    n = A.shape[0]

    if A.ndim == 2:     # in shape of [height, width] 
        for k in range(n):
            L[k, k] = np.sqrt(A[k, k])
            L[k+1:, k] = A[k+1:,  k] / L[k, k]
            for j in range(k + 1, n):
                A[j:, j] = A[j:, j] - L[j, k].conj().T * L[j:, k]
    elif A.ndim == 3:   #in shape of [height, width, channel]
        for k in range(n):
            L[k, k, :] = np.sqrt(A[k, k, :])
            L[k+1:, k, :] = A[k+1:,  k, :] / np.expand_dims(L[k, k, :], axis=0)
            for j in range(k + 1, n):
                A[j:, j, :] = A[j:, j, :] - np.expand_dims(L[j, k, :].conj().T, axis=0) * L[j:, k, :]
    else:
        raise NotImplementedError
    
    if dtype == 'torch':
        L = torch.from_numpy(L)

    return L


def wishart_noise(sigma, ENL: int=3):
    ''' Generate wishart noise, follow paper "Generation of Sample Complex Wishart Distributed Matrices and Change Detection in Polarimetric SAR Data"

    @in     -sigma      -original matix, in shape of [3, 3, len_]
    @in     -ENL        -equivalent number of looks
    @ret    
    '''
    h, w, len_ = sigma.shape
    if (h!=3) or (w!=3):
        raise ValueError('shape of a covariance matrix should be 3x3')
        
    c = my_cholesky(sigma)  # 3x3xlen_
    # generate complex gaussian distribution 
    x = np.random.randn(3, ENL, len_) + 1j*np.random.randn(3, ENL, len_)

    # x = x.reshape(3*ENL, len_)
    # np.set_printoptions(precision=3)
    # cov = np.cov(x, rowvar=False)
    # print(cov)
    # diff = np.abs(cov - np.eye(100))
    # print(f'max: {diff.max()}, min: {diff.min()}, mean: {diff.mean()}')

    w = mat_mul_dot(x.conj(), x)/2            
    w = mat_mul_dot(mat_mul_dot(c, w), c.conj())
    w /= ENL

    # w = (x.conj().T@x + y.conj().T@y - 1j*(x.conj().T@y-y.conj().T@x))/2
    # w = c.conj().T @ w @ c
    
    return w


def mat_mul_dot(a: ndarray, b: ndarray) -> ndarray:
    ''' a specified calculation of matrices, calulate matrix product on the first two axes, while remain the last axes unchanged
    @in     -a,b    -numpy array, both in [i, j, ...] shape 
    @ret    -c      -numpy array, in [i, i, ...] shape
    '''
    return np.einsum('ij..., kj...->ik...', a, b)


if __name__=='__main__':
    ''' test c32t3() '''
    path = r'data/PolSAR_building_det/data/GF3/anshou/20190223/C3/0'
    t3 = c32t3(path)
    write_t3('./tmp', t3)
    rgb = (rgb_by_t3(t3)*255).astype(np.uint8)
    cv2.imwrite(osp.join('./tmp', 'pauli.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(t3)

    ''' test mat_mul_dot() '''
    # c1, c2, c3 = [], [], []
    # for ii in range(10):
    #     a = np.random.rand(3, 4)
    #     b = np.random.rand(3, 4)
    #     c1.append(a @ b.T)
    #     c2.append(a)
    #     c3.append(b)
    # c1 = np.stack(c1, axis=2)
    # c2 = np.stack(c2, axis=2)
    # c3 = np.stack(c3, axis=2)
    # # c4 = np.einsum('ij..., kj...->ik...', c2, c3)
    # c4 = mat_mul_dot(c2, c3)
    # for ii in range(10):
    #     print(np.array_str(c1[:, :, ii], precision=12))
    #     print()
    #     print(np.array_str(c4[:, :, ii], precision=12))
    #     print(np.equal(c1[:, :, ii], c4[:, :, ii]))
    #     print('-'*50)
    # print(np.equal(c1, c4))
    # print('done')

    ''' test my_cholesky() '''
    # ll = []
    # ll_1 = []
    # for ii in range(10):
    #     a = np.random.randn(3,5) + 1j*np.random.randn(3,5)
    #     b = a @ a.conj().T
    #     ll.append(b)
    #     ll_1.append(my_cholesky(b))
    # ll_1 = np.stack(ll_1, axis=2)
    # ll_2 = np.stack(ll, axis=2)
    # ll_2 = my_cholesky(ll_2)
    # for ii in range(10):
    #     print(np.array_str(ll_1[:, :, ii], precision=4))
    #     print()
    #     print(np.array_str(ll_2[:, :, ii], precision=4))
    #     print('-'*50)
    # print('done')


    ''' test wishart_noise() '''
    a = np.random.randn(3,1) + 1j*np.random.randn(3,1)
    b = a @ a.conj().T
    c = wishart_noise(b, 7)
    print(c)


    ''' test read_s2() func '''
    # path = r'data/SAR_CD/GF3/data/E139_N35_日本横滨/降轨/1/20190615/s2'
    # path = r'./tmp/s2'
    # save_path = r'./tmp'
    # a = read_s2(path)
    # ps = rgb_by_s2(a)
    # print(cv2.imwrite(osp.join(save_path, 'pauli_s2.png'), cv2.cvtColor(np.transpose((255*ps), (1,0,2)).astype(np.uint8), cv2.COLOR_BGR2RGB)))


    ''' test split_patch_s2() func '''
    # path = r'/data/csl/SAR_CD/GF3/data'
    # for root, subdirs, files in os.walk(path):
    #     if 's2' == root[-2:]:
    #         split_patch_s2(root, transpose=True)
    ''' test split_patch() func '''


    ''' test Hokeman_decomposition() func '''
    # path = r'/home/csl/data/AIRSAR_Flevoland/C3/'
    # save_path = r'./tmp'
    # c3 = read_c3(path, out='save_space')
    # rgb = rgb_by_c3(c3)
    # rgb = (255*rgb).astype(np.uint8)
    # cv2.imwrite(osp.join(save_path, 'pauli.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # print('c3 max:', c3.max(), ' min:', c3.min())
    # h = Hokeman_decomposition(c3)
    # print('h max:', h.max(), ' min:', h.min())
    # for ii in range(9):
    #     # cv2.imwrite(osp.join(save_path, f'{ii}.png'), 255*min_max_map(h[ii, :, :]))
    #     tmp = h[ii, :, :]
    #     tmp = np.log(tmp)
    #     plt.hist(tmp)
    #     plt.savefig(osp.join(save_path, f'{ii}.png'))
    ''' test Hokeman_decomposition() func '''