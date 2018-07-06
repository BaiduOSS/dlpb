#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Description : This is for demo of broadcast in python numpy
    Authors     : tanzhongyi(tanzhongyi@baidu.com)
    Date        : 2017-12-18
"""
import numpy as np


def demo1():
    """
    不使用广播机制，完成对位的相加
    """

    v_a = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    v_b = np.array([
        [1, 1, 1],
        [1, 1, 1]
    ])

    print "1. 相同维度array, 进行对位运算, 应该是" \
          "array([[2, 3, 4],[5, 6, 7]])实际为："
    print v_a + v_b


def demo2():
    """
    横向扩展
    """

    v_a = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
    v_b = 2.0
    v_c = v_a * v_b

    print "\n2. 广播机制下, v_c应该为array([[ 2.],[ 4.],[ 6.]])，实际为"
    print v_c


def demo3():
    """
    纵向扩展
    """
    v_a = np.array([[0.0, 0.0, 0.0],
                    [10.0, 10.0, 10.0],
                    [20.0, 20.0, 20.0],
                    [30.0, 30.0, 30.0]])

    v_b = np.array([0, 1, 2])

    v_c = v_a + v_b

    print "\n3. 广播机制下,应该为\narray([[  0.,  1.,  2.],"
    print "[ 10.,  11.,  12.],"
    print "[ 20.,  21.,  22.],"
    print "[ 30.,  31.,  32.]])"
    print "，实际为"

    print v_c


def main():
    """
    the main function
    """
    # 第一个demo
    demo1()

    # 第二个demo
    demo2()

    # 第三个demo
    demo3()


if __name__ == '__main__':
    main()
