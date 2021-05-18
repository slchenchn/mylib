'''
Author: Shuailin Chen
Created Date: 2020-11-17
Last Modified: 2021-05-18
	content: 
'''
''' simplekml模块过于麻烦，还不如自己弄一个'''
''' 整个kml文件都可以作为一个 kml node '''
''' 最后修改： 2020/06/28'''

import numpy as np


class kml_node(object):
    def __init__(self, name, value = None):
        # 定义允许创建的 node 的名字，不完全，只是我暂时只会用到这些node，实际上也不应该在这个类里面定义这些，因为每创建一个node，
        # 都会有一个这样的 name_list，浪费空间，应该再弄一个新的，用于创建一个 kml 文件的类，但是因为我比较懒，而且可以把整个kml
        # 文件当作一个 kml node，因此就懒得改了
        self.name_list = ('kml', 'Placemark', 'name', 'LookAt', 'longitude', 'latitude', 'range', 
            'tilt', 'heading', 'Style', 'LineStyle', 'color', 'width', 'LineString', 'coordinates')
        if name in self.name_list:
            self._name = name
            self._value = value
            self._children = []     #子节点
        else:
            raise ValueError('name undifined')

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    # 添加子节点的方法
    def add_children(self, node, value=None):
        if isinstance(node, kml_node):
            self._children.append(node)
        else:
            nd = kml_node(node, value)
            self._children.append(nd)
            return nd

    # 将其值作为字符串返回
    def value_str(self):
        vstr = ''
        if isinstance(self._value, (list, np.ndarray)):     # numpy的二维数组也可以用两个 for in 得到所有数据
            # num = 0
            for tp in self._value:
                # num += 1
                for num in tp:
                    vstr += str(num)
                    vstr += ','
                vstr = vstr[:-1] + '\n'
            # if num==

            for num in self._value[0]:
                vstr += str(num)
                vstr += ','
            vstr = vstr[:-1] + '\n'

        else:
            vstr = str(self.value) + '\n'
        return vstr

    # 将节点的开头标记作为字符串返回
    def name_str_pre(self):
        return '<' + self._name + '>\n'

    # 将节点的结尾标记作为字符串返回
    def name_str_post(self):
        return '</' + self._name + '>\n'

    # 通过递归得到节点及其自己点的信息，可以用于写入 kml 文件
    def __str__(self):
        pstr = ''
        if self._children:
            pstr += self.name_str_pre()
            for nd in self._children:       # 有了子节点应该就不会有 value 属性了
                pstr += str(nd)
                # pstr += '\n'
            pstr += self.name_str_post()
        else:
            pstr += self.name_str_pre() + self.value_str() + self.name_str_post()
        return pstr

    # 保存为 kml 文件
    def save(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as fp:
            fp.write('<!-- ?xml version="1.0" encoding="UTF-8"? -->\n')
            fp.write(str(self))

if __name__ == "__main__":
    ''' 测试代码'''
    name = 'test'
    longitude = 50.611770
    latitude = 26.212633
    range = 250000.0     # in meters
    tilt = 0
    heading = 0
    coords = [(50.497257,26.389493,8000.0), (50.433107,26.085930,8000.0), (50.729055,26.035034,8000.0), (50.794015,26.338654,8000.0), (50.497257,26.389493,8000.0)]
    kml = kml_node('kml')
    # print(kml)
    plamrk = kml.add_children('Placemark')
    # print(kml)
    plamrk.add_children('name', name)
    # print(kml)
    lookat = plamrk.add_children('LookAt', None)
    print(kml)
    lookat.add_children('longitude', longitude)
    lookat.add_children('latitude', latitude)
    lookat.add_children('range', range)
    lookat.add_children('tilt', tilt)
    lookat.add_children('heading', heading)
    ls = plamrk.add_children('LineString')
    ls.add_children('coordinates', coords)
    print(kml)