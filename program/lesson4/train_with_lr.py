#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Description : This is a demo of using logistic classification
    Authors     : Jiahui Liu(2505774110@qq.com)
    Date        : 2017-12-18
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import utils


def main():
    """
    show dataset and the result of logistic regression
    """

    np.random.seed(1)

    # 加载数据
    train_X, train_Y, test_X, test_Y = utils.load_data_sets()

    print "1. show the data set"
    plt.scatter(test_X.T[:, 0], test_X.T[:, 1], c=test_Y, s=40,
                cmap=plt.cm.Spectral)
    plt.title("show the data set")
    plt.show()

    shape_X = train_X.shape
    shape_Y = train_Y.shape
    m = train_Y.shape[1]

    print 'The shape of X is: ' + str(shape_X)
    print 'The shape of Y is: ' + str(shape_Y)
    print 'I have m = %d training examples!' % (m)

    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(train_X.T, train_Y.T)

    print "2. show the result of logistic classification"
    utils.plot_decision_boundary(lambda x: clf.predict(x), train_X, train_Y)
    plt.title("Logistic Regression")
    plt.show()

    lr_predictions = clf.predict(train_X.T)
    print ('Accuracy of logistic regression: %d ' % float((np.dot(train_Y, lr_predictions)
         + np.dot(1 - train_Y, 1 - lr_predictions)) / float(train_Y.size) * 100)
           + '% ' + "(percentage of correctly labelled datapoints)")


if __name__ == '__main__':
    main()
