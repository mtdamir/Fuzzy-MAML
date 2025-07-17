import numpy as np
import os
import random
import tensorflow as tf

# Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) 
              for i, path in zip(labels, paths) 
              for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

# Network helpers
def conv_block(inp, cweight, bweight, norm_layer=None, activation=tf.nn.relu, max_pool_pad='VALID', reuse=False, scope=''):
    """Perform convolution, normalization, nonlinearity, and max pool"""
    stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]

    # انجام عملیات کانولوشن
    if max_pool_pad == 'VALID':
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight

    # اعمال نرمال‌سازی اگه وجود داشته باشه
    if norm_layer is not None:
        normed = norm_layer(conv_output)
    else:
        normed = conv_output

    # اعمال تابع فعال‌سازی
    if activation is not None:
        normed = activation(normed)

    # اعمال Max Pool اگه لازم باشه
    if max_pool_pad == 'VALID':
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)

    return normed

# Loss functions
@tf.function
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred - label))

@tf.function
def xent(pred, label):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred))