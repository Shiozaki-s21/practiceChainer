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
