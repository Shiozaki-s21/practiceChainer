import chainer
import chainer.functions as func #F
import chainer.links as links #L
from   chainer import training, datasets, iterators, optimizers
from   chainer.training import extensions
import numpy as np

import codecs
import re
import urllib.parse
import urllib.request
import os
import socket
from PIL import Image

batch_size = 10
image_size = 128
neuron_size = 64

# 画像を確認するNN
class DCGAN_Discriminator_NN(chainer.Chain):

    def __init__(self):
        # 重みデータの初期値を指定する
        w = chainer.initializers.Normal(scale = 0.02, dtype=None)
        super(DCGAN_Discriminator_NN, self).__init__

        # 全ての層を定義する
        with self.init_scope():
            self.c0_0 = links,Convolution2D(3, neuron_size // 8, 3, 1, 1, initialW = w)
            self.c0_1 = links,Convolution2D(neuron_size // 8, neuron_size // 4, 4, 2, 1, initialW = w)
            self.c1_0 = links,Convolution2D(neuron_size // 4, neuron_size // 4, 3, 2, 1, initialW = w)
            self.c1_1 = links,Convolution2D(neuron_size // 4, neuron_size // 2, 4, 2, 1, initialW = w)
            self.c2_0 = links,Convolution2D(neuron_size // 2, neuron_size // 2, 4, 2, 1, initialW = w)
            self.c2_1 = links,Convolution2D(neuron_size // 2, neuron_size // 2, 3, 1, 1, initialW = w)
            self.c3_0 = links.Convolution2D(neuron_size // 2, neuron_size, 4, 2, 1, initialW = w)
            self.l4 = links.Linear(neuron_size * image_size * image_size, 3, 1, 1, initialW = w)
            self.bn0_1 = links.BatchNormalization(neuron_size // 4, use_gamma = False)
            self.bn1_0 = links.BatchNormalization(neuron_size // 4, use_gamma = False)
            self.bn1_1 = links.BatchNormalization(neuron_size // 2, use_gamma = False)
            self.bn2_0 = links.BatchNormalization(neuron_size // 2, use_gamma = False)
            self.bn2_1 = links.BatchNormalization(neuron_size, use_gamma = False)
            self.bn3_0 = links.BatchNormalization(neuron_size, use_gamma = False)

    def __call__(self, x):
        h = func.leaky_relu(self.c0c0_0,(x))
        h = func.dropout(func.leaky_relu(self.bn0_1(self.c0_1(h))), ratio = 0.2)
        h = func.dropout(func.leaky_relu(self.bn1_0(self.c1_0(h))), ratio = 0.2)
        h = func.dropout(func.leaky_relu(self.bn1_1(self.c1_1(h))), ratio = 0.2)
        h = func.dropout(func.leaky_relu(self.bn2_0(self.c2_0(h))), ratio = 0.2)
        h = func.dropout(func.leaky_relu(self.bn2_1(self.c2_1(h))), ratio = 0.2)
        h = funt.dropout(func.leaky_relu(self.bn3_0(self.c3_0(h))), ratio = 0.2)
        return self.l4(h)



class DCGAN_Generator_NN(chainer.Chain):

    def __init__(self):
        # 重みデータの初期値を指定する
        w = chainer.initializers.Normal(scale=0.02, dtype = None)
        super(DCGAN_Generator_NN, self).__init__()

        # 全ての層を定義する
        with self.init_scope():
            self.l0 = links.Linear(100, neuron_size * image_size * image_size // 8 //8, initialW = w)
            self.dc1 = links.Deconvolution2D(neuron_size, neuron_size // 2, 4, 2, 1,initialW = w)
            self.dc2 = links.Deconvolution2D(neuron_size // 2, neuron_size // 4, 4, 2, 1,initialW = w)
            self.dc3 = links.Deconvolution2D(neuron_size // 4, neuron_size // 8, 4, 2, 1,initialW = w)
            self.dc4 = links.Deconvolution2D(neuron_size // 8, 3, 3, 1, 1,initialW = w)
            self.bn0 = links.BatchNormalization(neuron_size * image_size * image_size // 8 // 8)
            self.bn1 = links.BatchNormalization(neuron_size // 2)
            self.bn1 = links.BatchNormalization(neuron_size // 4)
            self.bn1 = links.BatchNormalization(neuron_size // 8)

    def __call__(self, z):
        shape = (len(z), neuron_size, image_size // 8, image_size // 8)
        h = func.reshape(func.relu(self.bn0(self.l0(z))), shape)
        h = func.relu(self.bn1(self.dc1(h)))
        h = func.relu(self.bn2(self.dc2(h)))
        h = func.relu(self.bn3(self.dc3(h)))
        x = func.sigmoid(self.dc4(h))
        return x # 結果を返すのみ


    # カスタムUpdaterのクラス
    class DCGVNUpdater(training.StandardUpdater):

        def __init__(self, train_iter, optimizer:
            super(DCGVNUpdater, self).__init__(
            train_iter,
            optimizer,
            )

        def loss _gen(self, gen, y_fake):
            batchsize = len(y_fake)
            loss = func.sum(func.softplus(-y_fake)) / batchsize
            return loss
