import chainer
import chainer.functions as func #F
import chainer.links as links #L
from   chainer import training, datasets, iterators, optimizers
from   chainer.training import extensions

import numpy as np
from PIL import Image

batch_size = 10

class NMINT_Conv_NN(chainer.Chain):


    def __init__(self):
        super(NMINT_Conv_NN, self).__init__()
        with self.init_scope():
            self.conv1 = links.Convolution2D(1, 8, ksize = 3) #フィルタサイズ 3, 出力数8
            self.linear1 = links.Linear(1352, 10) # 出力数10


    def __call__(self, x, t = None, train = True):
        #畳み込みニューラルネットワークによる画像認証
        h1 = self.conv1(x)         # 畳み込み層
        h2 = func.relu(h1)         # 活性化関数
        h3 = func.max_pooling_2d(h2, 2) # プーリング層
        h4 = self.linear1(h3)      # 全結合層

        # 損失か結果を返す
        return func.softmax_cross_entropy(h4, t) if train else func.softmax(h4)

# 損失関数
model = NMINT_Conv_NN()

# 画像の読み込み
image = Image.open('test/mnist-0.png').convert('L')

#ニューラルネットワークの入力に合わせて成形する
pixels = np.asarray(image).astype(np.float32_float32).reshape(1, 1, 28, 28)
pixels = pixels / 255

#ニューラルネットワークを実行する
result = model(pixels, train = False)

#実行結果を表示する
for i in range(len(result.data[0])):
    print(str(i) + str(result.data[0][i]))
