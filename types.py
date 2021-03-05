from typing import Union
from numpy import ndarray
import numpy as np
import pandas as pd
from pandas.core import frame
import yaml
from pandas.plotting import table
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
    plt.savefig('./tmp/pp.png', bbox_inches='tight')
    fig.tight_layout()
    return fig


if __name__=='__main__':

    ''' test flatten_dict_summarWriter() func '''
    # cfg = {'data': {'dataloader': 'SAR_CD_Hoekman', 'img_cols': 512, 'img_rows': 512, 'path': 'data/SAR_CD/RS2', 'train_split': 'train', 'val_split': 'test'}, 'model': {'arch': 'siam-diff', 'input_nbr': 9, 'label_nbr': 2}, 'training': {'batch_size': 8, 'clip': False,  'n_workers': 8,  'print_interval': 10, 'resume': 'runs/siam-diff_cross..._model.pkl', 'train_epoch': 6000}, 'weight': [0.01, 0.99]}
    # print(flatten_dict_summarWriter(cfg))

    ''' test dict2image() func '''
    # a = {'a':'1', 'b':'2', 'c': '3'}
    # print(pd.DataFrame.from_dict(a, orient='index', columns=['hhh']))
    cfg_path = r'/home/csl/code/PolSAR_CD/configs/tile.yml'
    with open(cfg_path) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    print(cfg, '\n')
    img = dict2fig(cfg)
    print(img)
