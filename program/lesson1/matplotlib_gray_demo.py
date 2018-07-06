#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Description : This is for demo of using matplotlib in python
    Authors     : tanzhongyi(tanzhongyi@baidu.com)
    Date        : 2017-12-18
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage


def rgb2gray(rgb):
    """
    图像转化为灰度图实现
    args:
        rgb: 彩色图像
    return:
        np.dot: 灰度图像
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def demo1():
    """
    read one dog's image and plot it to screen
    """

    plt.figure('A Little White Dog')
    little_dog_img = np.array(scipy.ndimage.imread('./little_white_dog.jpg',
                                                   flatten=False))
    plt.imshow(little_dog_img)
    plt.show()


def demo2():
    """
    show two plots
    one is the original dog
    the other call rgb2gray to make it gray
    """
    # Z是小白狗的照片，img0就是Z，img1是Z做了个简单的变换
    little_dog_img = np.array(scipy.ndimage.imread('./little_white_dog.jpg',
                                                   flatten=False))
    little_dog_img = rgb2gray(little_dog_img)
    img_0 = little_dog_img
    img_1 = 1 - little_dog_img

    # cmap指定为'gray'用来显示灰度图
    fig = plt.figure('Auto Normalized Visualization')
    ax0 = fig.add_subplot(121)
    ax0.imshow(img_0, cmap='gray')
    ax1 = fig.add_subplot(122)
    ax1.imshow(img_1, cmap='gray')
    plt.show()


def main():
    """
    matploblib demo to show pictures
    """
    demo1()
    demo2()


if __name__ == '__main__':
    main()
