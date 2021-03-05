'''
Author: Shuailin Chen
Created Date: 2020-11-17
Last Modified:2021-01-25
'''
import os
from os import path
import numpy as np 
from matplotlib import pyplot as plt 
import os.path as osp
import cv2
import bisect
from numpy import ndarray
from mylib import file_utils as fu
from typing import Union


bin_files = ['C11.bin', 'C12_real.bin', 'C12_imag.bin', 'C13_real.bin', 
            'C13_imag.bin', 'C22.bin', 'C23_real.bin', 'C23_imag.bin',
            'C33.bin',]

hdr_elements = ['samples', 'lines', 'byte order', 'data type', 'interleave']

data_type = ['uint8', 'int16', 'int32', 'float32', 'float64', 'uint16', 'uint32',
            'int64', 'uint64']


def check_c3_path(path:str)->str:
    '''check the path whether contains the c3 folder, if not, add it'''
    if path[-3:] != r'\C3' and path[-3:] != r'/C3' and (not osp.isfile(osp.join(path, 'config.txt'))):
        # print(path[-3:])
        path = os.path.join(path, 'C3')
    return path

def read_hdr(path:str)->dict:
    ''' read header file '''
    path = check_c3_path(path)
    meta_info = dict()
    with open(os.path.join(path, 'C11.bin.hdr'), 'r') as hdr:
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


def read_c3(path:str, out:str='complex_vector_6', meta_info=None, count=-1, offset=0, is_print=None)->np.ndarray:
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
            -offset     -The offset (in bytes) from the file��s current position. 
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


def write_config_hdr(path:str, config:Union[dict, list, tuple])->None:
    """
    write config.txt file and Cxx.hdr file
    @in     -path       -data path
            -config     -config information require for .bin.hdr file, in a dict format
    """
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

    with open(osp.join(path, 'config.txt'), 'w') as cfg:
        cfg.write('Nrow\n')     # nrow = lines, ncol = smaples
        cfg.write(lines)
        cfg.write('\n---------\nNcol\n')
        cfg.write(samples)
        cfg.write('\n---------\n')
        cfg.write('PolarCase\nmonostatic\n---------\nPolarType\nfull')
    
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


def write_c3(path:str, data:ndarray, config:dict=None, is_print=False):    
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

    # write config.txt
    if config is not None:
        write_config_hdr(path, config)

    # write binary files
    if (isinstance(config, dict) and config['data type'] == '4') or isinstance(config, (list, tuple)):
        data = as_format(data, out='save_space')
        for idx, bin in enumerate(bin_files):
            fullpath = osp.join(path, bin)
            file = data[idx, :, :]
            file.tofile(fullpath)
    else:
        raise NotImplementedError('data type is not float32')


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


def rgb_by_c3(data:np.ndarray, type:str='pauli')->np.ndarray:
    ''' @brief   -create the pseudo RGB image with covariance matrix
    @in      -data  -input polSAR data
    @in      -type  -'pauli' or 'sinclair'
    @out     -RGB data in [0, 255]
    undone
    '''
    type = type.lower()

    data = np.real(as_format(data, out='complex_vector_6'))
    R = np.zeros(data.shape[:2], dtype = np.float32)
    G = R.copy()
    B = R.copy()

    # compute orginal RGB components
    if type == 'pauli':
        # print('test')
        R = 0.5*(data[0, :, :]+data[5, :, :])-data[2, :, :]
        G = data[3, :, :]
        B = 0.5*(data[0, :, :]+data[5, :, :])+data[2, :, :]

    # print(R, '\n')
    # abs
    R = np.abs(R)
    G = np.abs(G)
    B = np.abs(B)

    # clip
    R[R<np.finfo(float).eps] = np.finfo(float).eps
    G[G<np.finfo(float).eps] = np.finfo(float).eps
    B[B<np.finfo(float).eps] = np.finfo(float).eps

    # print(R, '\n')
    # logarithm 
    R = 10*np.log10(R)
    G = 10*np.log10(G)
    B = 10*np.log10(B)
    
    # normalize
    # R = min_max_contrast_median_map(R[R!=10*np.log10(np.finfo(float).eps)])
    # G = min_max_contrast_median_map(G[G!=10*np.log10(np.finfo(float).eps)])
    # B = min_max_contrast_median_map(B[B!=10*np.log10(np.finfo(float).eps)])
    R = min_max_contrast_median_map(R)
    G = min_max_contrast_median_map(G)
    B = min_max_contrast_median_map(B)

    # print(R.shape, G.shape, B.shape)
    return np.stack((R, G, B), axis=2)


def min_max_contrast_median(data:np.ndarray):
    ''' @breif use the iterative method to get special min and max value
    @out    - min and max value in a tuple
    '''
    # remove nan and inf, vectorization
    data = data.reshape(1,-1)
    data = data[~(np.isnan(data) | np.isinf(data))]

    # iterative find the min and max value
    med = np.median(data)
    med1 = med.copy()       # the minimum value
    med2 = med.copy()       # the maximum value
    for ii in range(3):
        part_min = data[data<med1]
        if part_min.size>0:
            med1 = np.median(part_min)
        else:
            break
    for ii in range(3):
        part_max = data[data>med2]
        if part_max.size>0:
            med2 = np.median(part_max)
        else:
            break
    return med1, med2


def min_max_contrast_median_map(data:np.ndarray)->np.ndarray:
    '''
    @brief  -map all the elements of x into [0,1] using        
            min_max_contrast_median function
    @out    -the nomalized np.ndarray
    '''
    min, max = min_max_contrast_median(data[data != 10*np.log10(np.finfo(float).eps)])
    # print('ggg   ', min, max, 'ggg')
    return np.clip((data-min)/(max - min), a_min=0, a_max=1)


def min_max_map(x):
    '''''''''''''''''''''''''''''''''
    @brief  map all the elements of x into [0,1] using min max map
    @in     x   np.ndarray
    @out        np.ndarray
    '''''''''''''''''''''''''''''''''
    min = x.reshape(1,-1).min()
    max = x.reshape(1,-1).max()
    return (x-min)/(max-min)


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
        whole_img = whole_img.transpose((1, 0, 2))
        tmp = whole_config['lines']
        whole_config['lines'] = whole_config['samples']
        whole_config['samples'] = tmp
        write_c3(path, whole_data, whole_config, is_print=True)
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


def Hokeman_decomposition(data:ndarray)->ndarray:
    ''' calculate the Hokeman decomposition, which transforms the C3 matrix into 9 independent SAR intensities 
    @in     -data           -data to be transformed, in [channel, height, width] format
    @out    -H              -transformed Hoekman coefficient, in [channel, height, width] format
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
    data = as_format(data, 'save_space')
    data = data[[0,8,5,3,4,1,2,6,7], :, :]
    # data = data[[1,9,6,4,5,2,3,7,8], :, :]
    _, m, n = data.shape
    H = (vb @ data.reshape(9, -1)).reshape(9, m, n)
    H[H<np.finfo(float).eps] = np.finfo(float).eps
    return H


if __name__=='__main__':
    
    ''' test split_patch() func '''
    # path = r'/data/csl/SAR_CD/GF3/data'
    # for root, subdirs, files in os.walk(path):
    #     if 'C3' == root[-2:]:
            # split_patch(root)
    ''' test split_patch() func '''


    ''' test Hokeman_decomposition() func '''
    path = r'/home/csl/data/AIRSAR_Flevoland/C3/'
    save_path = r'./tmp'
    c3 = read_c3(path, out='save_space')
    rgb = rgb_by_c3(c3)
    rgb = (255*rgb).astype(np.uint8)
    cv2.imwrite(osp.join(save_path, 'pauli.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print('c3 max:', c3.max(), ' min:', c3.min())
    h = Hokeman_decomposition(c3)
    print('h max:', h.max(), ' min:', h.min())
    for ii in range(9):
        # cv2.imwrite(osp.join(save_path, f'{ii}.png'), 255*min_max_map(h[ii, :, :]))
        tmp = h[ii, :, :]
        tmp = np.log(tmp)
        plt.hist(tmp)
        plt.savefig(osp.join(save_path, f'{ii}.png'))

    ''' test Hokeman_decomposition() func '''