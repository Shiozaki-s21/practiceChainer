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

class SuperResolution_NN)(chainer.Chain):

    def __init__(self):
        #重みデータの初期値を指定する
        #FSRCNNでは重みの初期値を最適化しているため、手動で設定する
        w1 = chainer.initializers.Normal(scale = 0.0378, dtype = None)
        w2 = chainer.initializers.Normal(scale = 0.3536, dtype = None)
        w3 = chainer.initializers.Normal(scale = 0.1179, dtype = None)
        w4 = chainer.initializers.Normal(scale = 0.189,  dtype = None)
        w5 = chainer.initializers.Normal(scale = 0.0001, dtype = None)

        super(SuperResolution_NN, self).__init__()

        with self.init_scope():
            #ニューラルネットワークの層を作成
            self.c1 = links.Comvolution2D(1, 56, kseiz = 5, stride = 1, pad = 0, initalW = w1)
            self.l1 = links.PReLU()
            self.c2 = links.Comvolution2D(56, 12, kseiz = 1, stride = 1, pad = 0, initalW = w2)
            self.l2 = links.PReLU()
            self.c3 = links.Comvolution2D(12, 12, kseiz = 3, stride = 1, pad = 1, initalW = w3)
            self.l3 = links.PReLU()
            self.c4 = links.Comvolution2D(12, 12, kseiz = 3, stride = 1, pad = 1, initalW = w3)
            self.l4 = links.PReLU()
            self.c5 = links.Comvolution2D(12, 12, kseiz = 3, stride = 1, pad = 1, initalW = w4)
            self.l5 = links.PReLU()
            self.c6 = links.Comvolution2D(12, 12, kseiz = 3, stride = 1, pad = 1, initalW = w3)
            self.l6 = links.PReLU()
            self.c7 = links.Comvolution2D(12, 56, kseiz = 1, stride = 1, pad = 0, initalW = w4)
            self.l7 = links.PReLU()
            self.c8 = links.Comvolution2D(56, 1, kseiz = 1, stride = 1, pad = 0, initalW = w5)
