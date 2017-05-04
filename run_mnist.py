#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:26:51 2017

@author: herow
"""
import tensorflow as tf
import mnist_model as mnist
import tensorflow.examples.tutorials.mnist.input_data as input_data

data_set = input_data.read_data_sets('MNIST_data',one_hot=True)


with tf.Graph().as_default():
    images_placeholder = tf.placeholder("float",shape=[None,784])
    labels_placeholder = tf.placeholder("float",shape=[None,10])
    drop_out = tf.placeholder("float")
    learning_rate = tf.placeholder("float")
    predicts = mnist.inference(images_placeholder,drop_out)
    loss = mnist.loss(predicts,labels_placeholder)
    train_op= mnist.training(loss,learning_rate)
    correct_prediction = tf.equal(tf.argmax(predicts,1), tf.argmax(labels_placeholder,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in xrange(2000):
        batch = data_set.train.next_batch(50)
        
#        if i%100 == 0:
#            train_accuracy = sess.run(accuracy,feed_dict={images_placeholder:batch[0], 
#                     labels_placeholder:batch[1], drop_out:1.0, learning_rate:1e-4})
#            print "step %d, training accuracy %g"%(i,train_accuracy)
        sess.run(train_op,feed_dict={images_placeholder:batch[0], 
                     labels_placeholder:batch[1], drop_out:0.5, learning_rate:1e-4})
    test_accuracy = sess.run(accuracy,feed_dict={images_placeholder:data_set.test.images, 
                     labels_placeholder:data_set.test.labels, drop_out:1.0, learning_rate:1e-4})
    print "test accuracy %g"%test_accuracy