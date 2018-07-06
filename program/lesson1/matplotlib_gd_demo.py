#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Description : This is for demo of using matplotlib in gradient descent
    Authors     : tanzhongyi(tanzhongyi@baidu.com)
    Date        : 2017-12-18
"""
import matplotlib.pyplot as plt
import numpy as np


def square_func(v_x):
    """
    构建目标函数实现
    args:
        x: 自变量
    return:
        np.square(x): 目标函数
    """
    return np.square(v_x)


def derivative_func(v_x):
    """
    目标函数一阶导数也即是偏导数实现
    args:
        x: 目标函数
    return:
        2 * x: 目标函数一阶导数
    """
    return 2 * v_x


def gradient_descent(x_start, deri_func, iterations, learning_rate):
    """
    梯度下降法函数
    args:
        x_start: x的起始点
        deri_func: 目标函数的一阶导函数
        iterations: 迭代次数
        learning_rate: 学习率
        x在每次迭代后的位置（包括起始点），长度为iterations+1
    return:
        xs: 求在epochs次迭代后x的更新值
    """
    xs = np.zeros(iterations + 1)
    x = x_start
    xs[0] = x
    for i in range(iterations):
        dx = deri_func(x)
        x = x - dx * learning_rate
        xs[i + 1] = x
    return xs


def mat_plot():
    """
    using matplotlib to draw gradient descent graph
    """
    line_x = np.linspace(- 5, 5, 100)
    line_y = square_func(line_x)
    x_start = - 5
    iterations = 50
    learning_rate = 0.7
    x = gradient_descent(x_start, derivative_func, iterations, learning_rate)

    # 绘制二阶曲线
    plt.plot(line_x, line_y, c='b')

    # 绘制梯度下降过程的点
    plt.scatter(x, square_func(x), c='r', )
    plt.plot(x, square_func(x), c='r',
             label='learning rate={}'.format(learning_rate))

    # legend函数显示图例
    plt.legend()
    # show函数显示
    plt.show()


if __name__ == "__main__":
    mat_plot()
