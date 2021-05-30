'''
Author: Shuailin Chen
Created Date: 2021-04-21
Last Modified: 2021-05-30
	content: functions about general math
'''

import numpy as np
from numpy import ndarray

eps = np.finfo(float).eps

def var_with_known_mean(data: ndarray, mean: ndarray, axis, ddof: int) -> ndarray:
    ''' Calculate variance with known mean value 
    @in     -data       -Array containing numbers whose variance is desired
            -mean       -mean value of the array, should has the same dims with data
            -axis       -along which the var is performed
            -ddof       -0 for biased estimation, 1 for unbiased estimation
    '''
    assert (ddof==0 or ddof==1), r'ddof must in {0, 1}'
    x = np.abs(data-mean)**2
    return np.sum(x, axis=axis) / (np.prod(np.size(x)//np.size(mean))-ddof)


def check_inf_nan(data, prefix=None, warn=True):
    ''' Check if the data contains inf and nan values
    
    Args:
        data (ndarray): data to be examined
        prefix (str): prefix of the warning message. Default: None
        warn (bool): whether to raise a warning message. Default: True
    
    Retures:
        True if there exists inf or nan values, False otherwise
    '''

    num_nan = np.isnan(data).sum()
    num_inf = np.isinf(data).sum()

    check = False
    if num_nan > 0:
        check = True
        if warn:
            UserWarning(f'{prefix}: nan value exist')
    if num_inf > 0:
        check = True
        if warn:
            UserWarning(f'{prefix}: inf value exist')
        
    return check
    

def min_max_map(x, axis=None):
    ''' Map x into [0,1] using min max map along a given axis
    
    Args:
        axis (None or int or tuple of ints): same usage as np.max()
    '''
    # min = x.reshape(1,-1).min()
    # max = x.reshape(1,-1).max()
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    return (x-min)/(max-min)


def norm_3_sigma(data:np.ndarray, mean=None, std=None, type='complex'):
    ''' standardization, moreover, if the value beyonds mean+3*sigma, clip it 
    @in     -data       -PolSAR data, in CxHxWx... format
            -type       -'complex': calculate the complex mean and var value, 'abs': calculate the absolute mean and var value
    @ret    standardized data
    '''
    assert data.ndim>=3
    ret = data.copy()

    # absolute value of complex value
    if type=='abs':
        tmp = np.abs(ret)
    elif type=='complex':
        tmp = ret.copy()
    else:
        raise NotImplementedError

    if mean is None:
        mean = tmp.mean(axis=(-1, -2), keepdims=True)
    if std is None: 
        std = tmp.std(axis=(-1, -2), keepdims=True)
    ret /= (mean+3*std)
    ret_abs = np.abs(ret)
    ret_abs[ret_abs<1] = 1
    ret /= ret_abs

    return mean, std, ret    


def mat_mul_dot(a: ndarray, b: ndarray) -> ndarray:
    ''' A specified calculation of matrices, calulate matrix product on the first two axes, while remain the last axes unchanged
    @in     -a,b    -numpy array, both in [i, j, ...] shape 
    @ret    -c      -numpy array, in [i, i, ...] shape
    '''
    return np.einsum('ij..., kj...->ik...', a, b)

    

def min_max_contrast_median(data:np.ndarray, mask=None):
    ''' Use the iterative method to get special min and max value

    Args:
        data (ndaray): data to be processed
        mask (ndarray): mask for the valid pixel, 1 indicates valid, 0 
            indicates invalid, None indicats all valid. Default: None

    Returns:
        min and max value in a tuple
    '''
    
    if mask is not None:
        data = data[mask]

    # remove nan and inf, vectorization
    data = data.reshape(1,-1)
    data = data[~(np.isnan(data) | np.isinf(data))]

    # iterative find the min and max value
    med = np.median(data)
    med1 = med.copy()       # the minimum value
    med2 = med.copy()       # the maximum value
    for ii in range(3):
        part_min = data[data<med1]
        if len(part_min) > 0:
            med1 = np.median(part_min)
        else:
            break
    for ii in range(3):
        part_max = data[data>med2]
        if len(part_max) > 0:
            med2 = np.median(part_max)
        else:
            break
    return med1, med2


def min_max_contrast_median_map(data:np.ndarray, mask=None, is_print=False)->np.ndarray:
    '''Map all the elements of x into [0,1] using min_max_contrast_median function

    Args:
        data (ndarray): data to be mapped
        mask (ndarray): mask for the valid pixel, 1 indicates valid, 0 
            indicates invalid, None indicats all valid. Default: None
        is_print (bool): whether to print debug infos

    Returns:
        the nomalized np.ndarray
    '''
    min, max = min_max_contrast_median(data, mask=mask)
    if is_print:
        print(f'min: {min}, max: {max}')
    return np.clip((data-min)/(max - min), a_min=0, a_max=1)


if __name__=='__main__':
    ''' test min_max_map() '''
    a = np.array([[1, 4, 3], [2, 5, 8], [7, 6, 9]])
    # b = min_max_map(a)
    b = min_max_map(a, axis=1)
    print(f'before\n{a}\nafter\n{b}')


    ''' test var_with_known_mean() '''
    # a = np.random.randn(10, 20, 30)
    # axis = (1, 2)
    # ddof = 1
    # print(f'a: {a}\n')
    # b1 = np.var(a, axis=axis, ddof=ddof)
    # b2 = var_with_known_mean(a, np.mean(a, axis=axis, keepdims=True), axis=axis, ddof=ddof)
    # print(b1)
    # print()
    # print(b2)
    # print(f'\nerr:{np.abs(b1-b2)}')


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