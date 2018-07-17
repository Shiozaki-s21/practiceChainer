import chainer
import chainer.functions as func # F
import chainer.links as links # L
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

batch_size = 128

class SuperResolution_NN(chainer.Chain):

    def __init__(self):
        # 重みデータの初期値を指定する
        # FSRCNNでは重みの初期値を最適化しているため、手動で設定する
        w1 = chainer.initializers.Normal(scale = 0.0378, dtype = None)
        w2 = chainer.initializers.Normal(scale = 0.3536, dtype = None)
        w3 = chainer.initializers.Normal(scale = 0.1179, dtype = None)
        w4 = chainer.initializers.Normal(scale = 0.189,  dtype = None)
        w5 = chainer.initializers.Normal(scale = 0.0001, dtype = None)

        super(SuperResolution_NN, self).__init__()

        with self.init_scope():
            # ニューラルネットワークの層を作成
            self.c1 = links.Convolution2D(1, 56, ksize = 5, stride = 1, pad = 0, initialW = w1)
            self.l1 = links.PReLU()
            self.c2 = links.Convolution2D(56, 12, ksize = 1, stride = 1, pad = 0, initialW = w2)
            self.l2 = links.PReLU()
            self.c3 = links.Convolution2D(12, 12, ksize = 3, stride = 1, pad = 1, initialW = w3)
            self.l3 = links.PReLU()
            self.c4 = links.Convolution2D(12, 12, ksize = 3, stride = 1, pad = 1, initialW = w3)
            self.l4 = links.PReLU()
            self.c5 = links.Convolution2D(12, 12, ksize = 3, stride = 1, pad = 1, initialW = w4)
            self.l5 = links.PReLU()
            self.c6 = links.Convolution2D(12, 12, ksize = 3, stride = 1, pad = 1, initialW = w3)
            self.l6 = links.PReLU()
            self.c7 = links.Convolution2D(12, 56, ksize = 1, stride = 1, pad = 0, initialW = w4)
            self.l7 = links.PReLU()
            self.c8 = links.Convolution2D(56, 1, ksize = 1, stride = 1, pad = 0, initialW = w5)

    def __call__(self, x, t = None, train = True):
        h1 = self.l1(self.c1(x))
        h2 = self.l2(self.c2(h1))
        h3 = self.l3(self.c3(h2))
        h4 = self.l4(self.c4(h3))
        h5 = self.l5(self.c5(h4))
        h6 = self.l6(self.c6(h5))
        h7 = self.l7(self.c7(h6))
        h8 = self.c8(h7)
        # 損失か結果を返す
        return func.mean_squared_error(h8, t) if train else h8

# カスタムUpdaerクラス
class SRUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer):
        super(SRUpdater, self).__init__(
            train_iter,
            optimizer
        )

    def update_core(self):
        # 学習のためのコードを作成

        # データを１パッチ分取得
        batch = self.get_iterator('main').next()
        # Optimizerを取得
        optimizer = self.get_optimizer('main')

        # パッチ分のデータを作る
        x_batch = [] # 入力データ
        y_batch = [] # 正解データ

        for img in batch:
            # 高解像度データ
            hpix = np.array(img, dtype = np.float32) / 255.0
            y_batch.append([hpix[:,:,0]]) # Yのみの1chデータ

            # 低解像度データを作成する
            low = img.resize((16, 16), Image.NEAREST)
            lpix = np.array(low, dtype = np.float32) / 255.0
            x_batch.append([lpix[:,:,0]]) # Yのみの1chデータ

        # numpy of cupy 配列にする
        x = np.array(x_batch, dtype = np.float32)
        y = np.array(y_batch, dtype = np.float32)

        optimizer.update(optimizer.target, x, y)


# ニューラルネットワークの作成
model = SuperResolution_NN()

images = []

# 全てのファイル
fs = os.listdir('train')
for fn in fs:
    # 画像を読み込み
    img = Image.open('train/', + fn).resize((320, 320)).convert('YCbCr')
    cut_x = 0

    while cut_x <= 320 - 40:
        cur_y = 0
        while cur_y <= 320 - 40:
            # 画像から切り出し
            rect = (cut_x, cur_y, cur_x + 40, cur_y + 40)
            cropimg = img.crop(rect).copy()
            # 配列に追加
            images.append(cropimg)
            # 次の切り出し場所へ
            cur_y += 20
        cur_x += 20

# 繰り返し条件を作成する
train_iter = iterators.SerialIterator(images, batch_size, shuffle=True)

# 誤差逆伝播法アルゴリズムを選択する
optimizer = optimizers.Adam()
optimizer.setup(model)

# デバイスを選択してTrainerを作成する
updater = SRUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (10000, 'epoch'), out = "result")

# 学習の進展を表示する　
trainer.extend(extensions.ProgressBar())

# 中間結果を保存する
n_save = 0
@chainer.training.make_extension(trigger = (1000, 'epoch'))
def save_model(trainer):
    # NNのデータを保存
    global n_save
    n_save = n_save + 1
    chainer.serializers.save_hdf5('chapt03-' + str(n_save) + '.hdf5', model)
trainer.extend(save_model)

# 機械学習を実行する
trainer.run()

# 機械学習の結果を保存する
chainer.serializers.save_hdf5('chapt03.hdf5', model)
