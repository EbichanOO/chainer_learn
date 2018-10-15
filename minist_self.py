import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, utils, Variable
from chainer import datasets, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class MyNetwork(Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super(MyNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)
    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)

if __name__ == '__main__':
    from chainer.datasets import mnist
    train, test = mnist.get_mnist(withlabel=True, ndim = 1)

    '''
        import matplotlib.pyplot as plt
        x, t = train[0]
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.savefig('5.png')
        print('label:', t)
    '''
    from chainer import iterators
    batchsize = 128
    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

    model = MyNetwork()

    gpu_id = -1
    if gpu_id >= 0:
        model.to_gpu(gpu_id)
    