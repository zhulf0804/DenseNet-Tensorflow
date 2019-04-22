# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import input_data
import densenet as DenseNet

BATCH_SIZE = 100
HEIGHT = 32
WIDTH = 32
CHANNELS = 3
CLASSES = 10
saved_ckpt_path = './checkpoint/'

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, CHANNELS], name='x_input')
    y = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
    y_onehot = tf.one_hot(y, CLASSES, dtype=tf.float32)

logits = DenseNet.densenet_cifar(x, True)

with tf.name_scope('accuracy'):
    softmax = tf.nn.softmax(logits, axis=-1)
    correct_prediction = tf.equal(tf.argmax(y_onehot, 1), tf.argmax(softmax, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


test_data = input_data.read_test_data()

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    saver.restore(sess, './checkpoint/densenet.model-44000')

    #ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
    #if ckpt and ckpt.model_checkpoint_path:
    #    saver.restore(sess, ckpt.model_checkpoint_path)
    #    print("Model restored...")
    sum_accuracy = 0
    for i in range(100):
        test_img_data, test_labels = test_data.next_batch(BATCH_SIZE)
        test_accuracy = sess.run(accuracy, feed_dict={x:test_img_data, y:test_labels})
        sum_accuracy += test_accuracy
        print("on batch size: %d, test accuracy: %f" %(BATCH_SIZE, test_accuracy))

    print("The test set number is %d, the average accuracy is %f" %(100*100, sum_accuracy / 100))
