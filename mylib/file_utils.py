'''
Author: Shuailin Chen
Created Date: 2020-11-17
Last Modified: 2021-11-25
	content: to manipulate file and folde 
'''
import os
import os.path as osp
import errno

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
    ''' raad .txt file as a list, each item in list presents a row in the .txt file except the newline '''
    contents = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                contents.append(line)
    
    return contents


def write_file_from_list(data:list, filepath):
    ''' Write a list to a .txt file, each item in list presents a row in the .txt file '''
    with open(filepath, 'w') as f:
        for item in data:
            f.write(f'{item}\n')


def add_filename_suffix(filename, suffix):
    ''' Add a suffix to filename (right before the extension part) '''
    file_, ext = osp.splitext(filename)
    return file_ + suffix + ext


def replace_suffix(filename, suffix):
    ''' Replace the suffix of a filename '''
    file_, ext = osp.splitext(filename)
    return file_ + suffix


def force_symlink(file1, file2):
    ''' Create a symbolic link, if file2 exists, delete it first '''
    try:
        os.symlink(file1, file2)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(file2)
            os.symlink(file1, file2)
        else:
            raise e