import chainer
import chainer.functions as func #F
import chainer.links as links #L
from   chainer import training, datasets, iterators, optimizers
from   chainer.training import extensions
import numpy as np
import os
from numpy import random
from PIL import Image

batch_size = 10
image_size = 128
neuron_size = 64

# 画像を確認するNN
class DCGAN_Discriminator_NN(chainer.Chain):

    def __init__(self):
        # 重みデータの初期値を指定する
        w = chainer.initializers.Normal(scale = 0.02, dtype=None)
        super(DCGAN_Discriminator_NN, self).__init__()

        # 全ての層を定義する
        with self.init_scope():
            self.c0_0 = links.Convolution2D(3, neuron_size // 8, 3, 1, 1, initialW = w)
            self.c0_1 = links.Convolution2D(neuron_size // 8, neuron_size // 4, 4, 2, 1, initialW = w)
            self.c1_0 = links.Convolution2D(neuron_size // 4, neuron_size // 4, 3, 2, 1, initialW = w)
            self.c1_1 = links.Convolution2D(neuron_size // 4, neuron_size // 2, 4, 2, 1, initialW = w)
            self.c2_0 = links.Convolution2D(neuron_size // 2, neuron_size // 2, 4, 2, 1, initialW = w)
            self.c2_1 = links.Convolution2D(neuron_size // 2, neuron_size // 2, 3, 1, 1, initialW = w)
            self.c3_0 = links.Convolution2D(neuron_size // 2, neuron_size, 4, 2, 1, initialW = w)
            self.l4 = links.Linear(neuron_size * image_size * image_size // 8 // 8, 1, initialW = w)
            self.bn0_1 = links.BatchNormalization(neuron_size // 4, use_gamma = False)
            self.bn1_0 = links.BatchNormalization(neuron_size // 4, use_gamma = False)
            self.bn1_1 = links.BatchNormalization(neuron_size // 2, use_gamma = False)
            self.bn2_0 = links.BatchNormalization(neuron_size // 2, use_gamma = False)
            self.bn2_1 = links.BatchNormalization(neuron_size, use_gamma = False)
            self.bn3_0 = links.BatchNormalization(neuron_size, use_gamma = False)

    def __call__(self, x):
        h = func.leaky_relu(self.c0_0,(x))
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
            self.bn2 = links.BatchNormalization(neuron_size // 4)
            self.bn3 = links.BatchNormalization(neuron_size // 8)

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

    def __init__(self, train_iter, optimizer):
        super(DCGVNUpdater, self).__init__(
        train_iter,
        optimizer)


    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = func.sum(func.softplus(-y_fake)) / batchsize
        return loss


    def loss_diss(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = func.sum(func.softplus(-y_real)) / batchsize
        L2 = func.sum(func.softplus(y_fake)) / batchsize
        loss = L1 + L2
        return loss

    def update_core(self):
        # iteratorからパッチ分のデータを取得
        batch = self.get_iterator('main').next()
        src = self.converter(batch)

        # optimizerを取得
        optimizer_gen = self.get_optimizer('opt_gen')
        optimizer_dis = self.get_optimizer('opt_dis')
        # ニューラルネットワークのモデルを取得
        gen = optimizer_gen.target
        dis = optimizer_dis.target

        # 乱数データを取得
        rnd = random.uniform(-1, 1, (src.shape[0], 100))
        rnd = np.array(rnd, dtype = np.float32)

        # 画像を生成して認識と教師データから認識
        x_fake = gen(rnd)    # 乱数からの生成結果
        y_fake = dis(x_fake) # 乱数から生成したものを認識した結果
        y_real = dis(src)    # 教師データからの認識結果

        # ニューラルネットワークを学習　
        optimizer_dis.update(self.loss_diss, dis, y_fake, y_real)
        optimizer_gen.update(self.loss_gen, gen, y_fake)

# ニューラルネットワークを作成
model_gen = DCGAN_Generator_NN()
model_dis = DCGAN_Discriminator_NN()

images = []

fs = os.listdir('train')
for fn in fs:
    # 画像を読み込んで128 X 128ピクセルにリサイズ
    img = Image.open('train/' + fn).convert('RGB').resize((128, 128))
    # 画素データを0〜１の領域にする
    hpix = np.array(img, dtype = np.float32) / 255.0
    hpix = hpix.transpose()
    # 配列に追加
    images.append(hpix)

# 繰り返し条件を作成する
train_iter = iterators.SerialIterator(images, batch_size, shuffle=True)

# 誤差逆伝播法アルゴリズムを選択する
optimizer_gen = optimizers.Adam(alpha = 0.002, beta1 = 0.5)
optimizer_gen.setup(model_gen)
optimizer_dis = optimizers.Adam(alpha = 0.002, beta1 = 0.5)
optimizer_dis.setup(model_dis)

# デバイスを選択してtrainerを作成する
updater = DCGVNUpdater(train_iter, \
{'opt_gen':optimizer_gen, 'opt_dis':optimizer_dis})

trainer = training.Trainer(updater, (10000, 'epoch'), out = "result")
# 学習の進展を表示するようにする
trainer.extend(extensions.ProgressBar())

# 中間結果を保存する
n_save = 0
@chainer.training.make_extension(trigger=(1000, 'epoch'))
def save_model(trainer):
    # NNのデータを保存
    global n_save
    n_save = n_save + 1
    chainer.serializers.save_hdf5('chapt04-gen-' + str(n_save) + '.hdf5', model_gen)
    chainer.serializers.save_hdf5('chapt04-dis-' + str(n_save) + '.hdf5', model_dis)
trainer.extend(save_model)

# 機械学習を実行する
trainer.run()

# 学習結果を保存する
chainer.serializers.save_hdf5('chapt04.hdf5', model_gen)
