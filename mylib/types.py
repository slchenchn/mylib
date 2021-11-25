'''
Author: Shuailin Chen
Created Date: 2020-11-17
Last Modified: 2021-11-25
	content: manipulate some basic types of python
'''
import random

import torch
from typing import Union
from numpy import ndarray
import numpy as np
import pandas as pd
from pandas.core import frame
import yaml
from pandas.plotting import table
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from colorama import Fore
import re


LINE_SEPARATOR = '-'*100

def rreplace(s, old, new, occurrence):
    ''' Replace a string from the right '''
    assert isinstance(s, str)
    li = s.rsplit(old, occurrence)
    return new.join(li)

def re_find_only_one(pattern, string):
    ''' find only one match '''
    match = re.findall(pattern, string)
    assert len(match)==1, f'expect to get only one match, got {len(match)}'
    return match[0]
    

def print_separate_line(message=None, color=None):
    ''' Print separate line with message '''
    if message is None:
        str_ = f'\n{color}{LINE_SEPARATOR}{Fore.RESET}\n'
    else:
        str_ = f'\n{color}{LINE_SEPARATOR}\n{message}\n{LINE_SEPARATOR}{Fore.RESET}\n'

    print(str_)


# def nest_dict(input:dict, sep='.', num=0)->dict:
#     ''' Nest a nested dict '''

#     if not isinstance(input, dict):
#         raise TypeError('the input param should be a dict')
    
#     out = {}

#     for k, v in input.items():
#         if not '.' in k:
#             out = input
#         else:
#             sub_key1, subkey2 = k.split(sep, 1)
#             # out[sub_key1] = nest_dict({subkey2:v}, sep=sep, num=1)
#             out.add({sub_key1:nest_dict(
#                                 {subkey2:v}, sep=sep, num=1)
#                                 })
#             if num == 0:
#                 print(out)
    
#     # print(out)
#     return out


def flatten_dict(input:dict, parent_key='', sep='.')->dict:
    ''' flatten a nested dict '''

    if not isinstance(input, dict):
        raise TypeError('the input param should be a dict')
    else:
        out = {}
        for k, v in input.items():
            new_key = parent_key + sep + k if parent_key else k
            if not isinstance(v, dict):
                out.update({new_key: v})
            else:
                out.update(flatten_dict(v, new_key))
    return out


def flatten_dict_summarWriter(input:dict)->dict:
    ''' flatten a nested dict, and change list-like object to str, for the puspose of summarywrite, '''
    flatted_dict = flatten_dict(input)
    for k, v in flatted_dict.items():
        if not isinstance(v, (bool, str, float, int)):
            if isinstance(v, (tuple, list)):
                flatted_dict.update({k:list2str(v)})
    return flatted_dict


def list2str(input:list)->str:
    ''' change list items to str type '''
    s = ''
    for e in input:
        s += str(e)
        s += ', '
    return s[:-2]


def dict2fig(d:dict)->Figure:
    d = flatten_dict_summarWriter(d)
    keys, vals = [], []
    for k, v in d.items():
        keys.append(k)
        vals.append([v])
    # tabs = list(d.items())
    # rows = len(tabs)
    rows = len(keys)
    fig, ax = plt.subplots()
    plt.axis('off')
    ax.table(cellText=vals, loc='center', cellLoc='center', rowLabels=keys, rowColours=['C1']*rows)
    # ax.table(cellText=vals,   rowLabels=keys)
    # plt.savefig('./tmp/pp.png', bbox_inches='tight')
    fig.tight_layout()
    return fig


def list_pop(list_, idx):
    ''' Pop a list with list-like indices 

    Args:
        list_ (list): a list object.
        idx (int or tupe or list): indices
    
    Returns:
        popped (list): popped objects
        new_list (list): popped list
    '''

    if not isinstance(idx, (tuple, list)):
        idx = [idx]

    popped = [list_[ii] for ii in idx]
    new_list = [j for i,j in enumerate(list_) if i not in idx]

    return popped, new_list


def list_numpy_to_torch(list_):
    ''' Convert a list of ndarray to torch tensor

    Args:
        list_ (list): list of numpy ndarray

    Return:
        list of torch tensor
    '''

    if isinstance(list_, (tuple, list)):
        newlist = [torch.from_numpy(ii) for ii in list_]
        return newlist
    else:
        raise ValueError('Wrong data type')


if __name__=='__main__':

    ''' test list_numpy_to_torch() '''
    a = np.array([1,2,3])
    b = np.array([[1],[5]])
    c = [a, b]
    print(c)
    d = list_numpy_to_torch(a)
    print('\n', d)


    ''' test list_pop() '''
    # a = list(range(10))
    # idx = random.sample(range(10), 5)
    # # idx = 7
    # print(f'array: {a}\nidx: {idx}')
    # p, n = list_pop(a, idx)
    # print(f'popped: {p}, \nnew list:{n}')
    # # print(a)



    ''' test flatten_dict_summarWriter() func '''
    # cfg = {'data': {'dataloader': 'SAR_CD_Hoekman', 'img_cols': 512, 'img_rows': 512, 'path': 'data/SAR_CD/RS2', 'train_split': 'train', 'val_split': 'test'}, 'model': {'arch': 'siam-diff', 'input_nbr': 9, 'label_nbr': 2}, 'training': {'batch_size': 8, 'clip': False,  'n_workers': 8,  'print_interval': 10, 'resume': 'runs/siam-diff_cross..._model.pkl', 'train_epoch': 6000}, 'weight': [0.01, 0.99]}
    # print(flatten_dict_summarWriter(cfg))

    ''' test dict2image() func '''
    # # a = {'a':'1', 'b':'2', 'c': '3'}
    # # print(pd.DataFrame.from_dict(a, orient='index', columns=['hhh']))
    # cfg_path = r'/home/csl/code/PolSAR_CD/configs/tile.yml'
    # with open(cfg_path) as fp:
    #     cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # print(cfg, '\n')
    # img = dict2fig(cfg)
    # print(img)

    print('done')
