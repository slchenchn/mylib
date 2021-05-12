'''
Author: Shuailin Chen
Created Date: 2021-04-21
Last Modified: 2021-04-27
	content: 
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
    

if __name__=='__main__':
    ''' test var_with_known_mean() '''
    a = np.random.randn(10, 20, 30)
    axis = (1, 2)
    ddof = 1
    print(f'a: {a}\n')
    b1 = np.var(a, axis=axis, ddof=ddof)
    b2 = var_with_known_mean(a, np.mean(a, axis=axis, keepdims=True), axis=axis, ddof=ddof)
    print(b1)
    print()
    print(b2)
    print(f'\nerr:{np.abs(b1-b2)}')
