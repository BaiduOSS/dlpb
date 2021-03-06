#!/usr/bin/env python
# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import os
from PIL import Image
import numpy as np
import paddle.v2 as paddle

with_gpu = os.getenv('WITH_GPU', '0') != '0'


def softmax_regression(img):
    predict = paddle.layer.fc(
        input=img, size=10, act=paddle.activation.Softmax())
    return predict


def multilayer_perceptron(img):
    # The first fully-connected layer
    hidden1 = paddle.layer.fc(input=img, size=128, act=paddle.activation.Relu())
    # The second fully-connected layer and the according activation function
    hidden2 = paddle.layer.fc(
        input=hidden1, size=64, act=paddle.activation.Relu())
    # The thrid fully-connected layer, note that the hidden size should be 10,
    # which is the number of unique digits
    predict = paddle.layer.fc(
        input=hidden2, size=10, act=paddle.activation.Softmax())
    return predict


def convolutional_neural_network(img):
    # first conv layer
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # second conv layer
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # fully-connected layer
    predict = paddle.layer.fc(
        input=conv_pool_2, size=10, act=paddle.activation.Softmax())
    return predict


def plot_costs(costs):

    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = 0.00002")
    # plt.show()
    plt.savefig('costs.png')


def main():
    paddle.init(use_gpu=with_gpu, trainer_count=1)

    # define network topology
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))

    # Here we can build the prediction network in different ways. Please
    # choose one by uncomment corresponding line.
    predict = softmax_regression(images)
    # predict = multilayer_perceptron(images)
    # predict = convolutional_neural_network(images)

    cost = paddle.layer.classification_cost(input=predict, label=label)

    parameters = paddle.parameters.create(cost)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.01 / 128.0,
        momentum=0,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    lists = []
    costs = []

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                costs.append(event.cost)

                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                trainer.save_parameter_to_tar(f)

            result = trainer.test(reader=paddle.batch(
                paddle.dataset.mnist.test(), batch_size=128))
            print "Test with Pass %d, Cost %f, %s\n" % (
                event.pass_id, result.cost, result.metrics)
            lists.append((event.pass_id, result.cost,
                          result.metrics['classification_error_evaluator']))

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=128),
        event_handler=event_handler,
        num_passes=5)

    # find the best pass
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
    print 'The classification accuracy is %.2f%%' % (100 - float(best[2]) * 100)

    #
    plot_costs(costs)


if __name__ == '__main__':
    main()
