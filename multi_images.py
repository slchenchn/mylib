'''
Author: Shuailin Chen
Created Date: 2021-06-21
Last Modified: 2021-09-19
	content: 
'''

import os
import os.path as osp

import numpy as np
from torch import Tensor
import torch

from mylib import image_utils as iu


def split_batches(x: Tensor):
    ''' Split a 2*B batch of images into two B images per batch, in order to adapt to MMsegmentation '''

    assert x.ndim == 4, f'expect to have 4 dimensions, but got {x.ndim}'
    batch_size = x.shape[0] // 2
    x1 = x[0:batch_size, ...]
    x2 = x[batch_size: , ...]
    return x1, x2


def merge_batches(x1: Tensor, x2: Tensor):
    ''' merge two batchs each contains B images into a 2*B batch of images in order to adapt to MMsegmentation '''

    assert x1.ndim == 4 and x2.ndim == 4,   f'expect x1 and x2 to have 4 \
                dimensions, but got x1.dim: {x1.ndim}, x2.dim: {x2.ndim}'
    return torch.cat((x1, x2), dim=0)


def split_images(x: Tensor):
    ''' Split a 2*c channels image into two c channels images, in order to adapt to MMsegmentation '''

    # determine 3D tensor or 4D tensor
    if x.ndim == 4:
        channels = x.shape[1]
        channels //= 2
        x1 = x[:, 0:channels, :, :]
        x2 = x[:, channels:, :, :]
    elif x.ndim == 3:
        channels = x.shape[-1]
        channels //= 2
        x1 = x[..., 0:channels]
        x2 = x[..., channels: ]
    else:
        raise ValueError(f'dimension of x should be 3 or 4, but got {x.ndim}')
        
    return x1, x2
