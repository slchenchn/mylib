from pathlib import Path
import os
import pickle
''' 获得具有层次的文件夹结构，以 dict 的形式表示 '''

def fs_tree_to_dict(path_):
    file_token = ''
    for root, dirs, files in os.walk(path_):
        tree = {d: fs_tree_to_dict(os.path.join(root, d)) for d in dirs}
        tree.update({f: file_token for f in files})
        return tree  # note we discontinue iteration trough os.walk



if __name__ == '__main__':
    tree = fs_tree_to_dict(r'C:\Users\wh\Documents\Tencent Files\1106982578\FileRecv')
    print(tree)
    with open('foldertree.p', 'wb') as f:
        pickle.dump(tree, f) 