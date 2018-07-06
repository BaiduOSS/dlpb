#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Description : This is for demo of using vector transpose in python numpy
    Authors     : tanzhongyi(tanzhongyi@baidu.com)
    Date        : 2017-12-18
"""

import numpy as np


def main():
    """
    show matrix's transpose
    """

    # 初始化矩阵matrix_x, matrix_x为
    # [[ 0  1  2  3]
    # [ 4  5  6  7]
    # [ 8  9 10 11]]

    matrix_x = np.array(np.arange(12).reshape((3, 4)))
    print "原始矩阵：\n" + str(matrix_x)
    print "shape 为" + str(matrix_x.shape)

    # 利用函数transpose()实现转置，转置后为矩阵t：
    # [[ 0  4  8]
    # [ 1  5  9]
    # [ 2  6 10]
    # [ 3  7 11]]
    # Your Code Begin：
    matrix_y = matrix_x
    # Your Code End：
    print "\n转置后矩阵：\n" + str(matrix_y)
    print "shape 为" + str(matrix_y.shape)


if __name__ == '__main__':
    main()
