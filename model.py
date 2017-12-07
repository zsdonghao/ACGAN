#! /usr/bin/python
# -*- coding: utf8 -*-
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def G(z, c, is_train=False, reuse=False, batch_size=64):                # ref: https://github.com/zsdonghao/dcgan/blob/master/model.py
    image_size = 32
    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16)
    gf_dim = 64
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        nz = InputLayer(z, name='inz')
        nc = EmbeddingInputlayer(c,
                vocabulary_size = 10,
                embedding_size = 10,
                name='inc')
        nc = FlattenLayer(nc, 'flat')
        n = ConcatLayer([nz, nc], 1, name='concat')

        n = DenseLayer(n, n_units=gf_dim*8*s8*s8, name='d1')
        n = ReshapeLayer(n, [-1, s8, s8, gf_dim*8], name='reshape')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, name='bn1')
        # print(n.outputs)
        # exit()

        n = DeConv2d(n, gf_dim*2, (3, 3), out_size=(s4, s4), strides=(2, 2),
                batch_size=batch_size, padding='SAME', name='de1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, name='dbn1')

        n = DeConv2d(n, gf_dim, (3, 3), out_size=(s2, s2), strides=(2, 2),
                batch_size=batch_size, padding='SAME', name='de2')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, name='dbn2')

        n = DeConv2d(n, 3, (3, 3), out_size=(image_size, image_size),
                batch_size=batch_size, strides=(2, 2), act=tf.nn.tanh, padding='SAME', name='do')
        return n

def D(x, is_train=False, reuse=False):
    image_size = 32
    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16)
    df_dim = 64
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        n = InputLayer(x, name='in')
        n = Conv2d(n, df_dim, (3, 3), (2, 2), act=lrelu, padding='SAME', name='c0')

        n = Conv2d(n, df_dim*2, (3, 3), (2, 2), padding='SAME', name='c1')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, name='bn1')

        n = Conv2d(n, df_dim*4, (3, 3), (2, 2), padding='SAME', name='c2')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, name='bn2')

        # n = Conv2d(n, df_dim*8, (5, 5), (2, 2), act=None,
        #         padding='SAME', W_init=w_init, name='d/h3/conv2d')
        # n = BatchNormLayer(n, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d/h3/batch_norm')

        n = FlattenLayer(n, name='flat')
        n1 = DenseLayer(n, n_units=1 , name='real/fake')
        n2 = DenseLayer(n, n_units=10, name='class')
        n1 = tl.layers.merge_networks([n1, n2])
    return n1, n1.outputs, n2.outputs
