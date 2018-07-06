#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Description : This is for demo of using matplot lib to plot data
    Authors     : tanzhongyi(tanzhongyi@baidu.com)
    Date        : 2017-12-18
"""

import numpy as np
import matplotlib.pyplot as plt


def main(param_a, param_b):
    """
    展示房屋价格与房屋面积的分布以及参数param_a，param_b生成的线性回归结果
    Args:
        param_a: 线性回归拟合结果，斜率a
        param_b: 线性回归拟合结果，截距b
    Return:
    """
    data = np.loadtxt('data.txt', delimiter=',')

    x_area = data[:, 0]
    y_price = data[:, 1]
    y_predict = x_area * param_a + param_b
    plt.scatter(x_area, y_price, marker='.', c='r', label='True')
    plt.title('House Price Distributions of XXX City XXX Area')
    plt.xlabel('House Area ')
    plt.ylabel('House Price ')
    plt.xlim(0, 250)
    plt.ylim(0, 2500)
    plt.plot(x_area, y_predict, label='Predict')
    plt.legend(loc='upper left')
    plt.savefig('result.png')
    plt.show()


if __name__ == '__main__':
    main(7.1, -62.3)
