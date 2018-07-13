import chainer
import chainer.functions as func #F
import chainer.links as links #L
from   chainer import training, datasets, iterators, optimizers
from   chainer.training import extensions
import numpy as np

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
# データセットの取得
# ndmi:データの次元数(縦×横×色)
train, test = chainer.datasets.get_mnist(ndim = 3)

# イテレータの作成
# 教師用のイテレータ
train_iter = iterators.SerialIterator(train, batch_size, shuffle = True)
test_iter = iterators.SerialIterator(test, 1, repeat = False, shuffle = False)

# トレーナーの作成
optimizer = optimizers.Adam()
optimizer.setup(model)

# デバイスを選択してTrainerを作成する
updater = training.StandardUpdater(train_iter, optimizer, device = None)
trainer = training.Trainer(updater, (5, 'epoch'), out = "result")

# テストをTrainerに設定する
trainer.extend(extensions.Evaluator(test_iter, model, device = None))

# 学習の進展を表示するようにする
trainer.extend(extensions.ProgressBar())

# 教師データとテスト用データの正解率を表示する
# 起動しない、要チェック
trainer.extend(extensions.LogReport(),
trainer.extend(extensions.PrintReport( entries=['epoch', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time' ])))

# 機械学習の実行
trainer.run()

# 学習結果を保存
chainer.serializers.save_hdf5('chapt02.hdf5', model)
