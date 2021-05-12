'''
Author: Shuailin Chen
Created Date: 2020-11-17
Last Modified: 2021-04-22
	content: 
'''
import os
import os.path as osp

def mkdir_if_not_exist(path):  
    """Make a directory if it does not exist."""
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)


def fs_tree_to_dict(path_):
    ''' expolre the folder structure and represent a dict '''
    file_token = ''
    for root, dirs, files in os.walk(path_):
        tree = {d: fs_tree_to_dict(os.path.join(root, d)) for d in dirs}
        tree.update({f: file_token for f in files})
        return tree  # note we discontinue iteration trough os.walk


def read_file_as_list(path):
    ''' raad .txt file as a list, each item in list presents a row in the .txt file '''
    contents = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                contents.append(line)
    
    return contents