#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Description : This is for demo of vectorization in python
    Authors     : tanzhongyi(tanzhongyi@baidu.com)
    Date        : 2017-12-18
"""
import time
import numpy as np


def main():
    """
    show vectorization improve performance
    """
    # 初始化两个1000000维的随机向量v1,v2用于矩阵相乘计算
    v_1 = np.random.rand(1000000)
    v_2 = np.random.rand(1000000)
    v_3 = 0

    # 矩阵相乘-非向量化版本
    tic = time.time()
    for i in range(1000000):
        v_3 = v_3 + v_1[i] * v_2[i]
    toc = time.time()
    print "1. 非向量化计算的执行结果如下"
    print "执行结果：" + str(v_3)
    print "执行时间：" + str((toc - tic) * 1000) + "ms" + "\n"

    # 矩阵相乘-向量化版本
    tic = time.time()
    # Your Code Begin：
    v_4 = v_3
    # Your Code End：
    toc = time.time()
    print "2. 向量化计算的执行结果如下"
    print "执行结果：" + str(v_4)
    print "执行时间：" + str((toc - tic) * 1000) + "ms"


if __name__ == '__main__':
    main()
