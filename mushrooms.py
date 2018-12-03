import chainer as ch
import chainer.functions as F
import chainer.links as L
from chainer import datasets
from chainer import training
from chainer.training import extensions
import numpy as np

def MLP(n_units, n_out):
    layer = ch.Sequential(L.Linear(n_units), F.relu)
    model = layer.repeat(2)
    model.append(L.Linear(n_out))
    return model

import matplotlib
matplotlib.use('Agg')

datafile = 'mushrooms.csv'
data_array = np.genfromtxt(
    datafile, delimiter=',', dtype=str, skip_header=1)
for col in range(data_array.shape[1]):
    data_array[:, col] = np.unique(data_array[:, col], return_inverse=True)[1]

X = data_array[:, 1:].astype(np.float32)
Y = data_array[:, 0].astype(np.int32)[:, None]
train, test = datasets.split_dataset_random(
    datasets.TupleDataset(X, Y), int(data_array.shape[0]*.7))

train_iter = ch.iterators.SerialIterator(train, 100)
test_iter = ch.iterators.SerialIterator(
    test, 100, repeat=False, shuffle=False)

model = L.Classifier(
    MLP(44, 1), lossfun=F.sigmoid_cross_entropy, accfun=F.binary_accuracy)

optimizer = ch.optimizers.SGD().setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=-1)
#updateしたら50 epochに達するまで回す + epochあたりの学習レートを出す
trainer = training.Trainer(updater, (50, 'epoch'), out='result')
#testを利用した評価をする
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
#グラフを作成する
trainer.extend(extensions.dump_graph('main/loss'))
#20 epochごとにオブジェクトのスナップショットをとるらしい
trainer.extend(extensions.snapshot(), trigger=(20, 'epoch'))
#各epochの評価統計情報ログ出す
trainer.extend(extensions.LogReport())
#2つのプロット画像を保存する
if extensions.PlotReport.available():
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'], 
        'epoch', file_name='accuracy.png'))
#ログをcmdに出す
trainer.extend(extensions.PlotReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

#run
trainer.run()
#use
x, t = test[np.random.randint(len(test))]

predict = model.predictor(x[None]).data
predict = predict[0][0]
if predict >= 0:
    print('Predicted Poisonous, Actual '+['Edible', 'Poisonous'][t[0]])
else:
    print('Predicted Edible, Actual '+['Edible', 'Poisonous'][t[0]])