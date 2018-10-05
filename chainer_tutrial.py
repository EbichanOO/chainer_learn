import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extension

class simpleMaths:
    def __init__(self):
        x = Variable(np.array([5], dtype=np.float32))
        self.centerBack(x)
    def Calculation(self, x):
        y = x**2 - 2*x + 1
        print(y.array)
    def Backward(self):
        y.backward()
        print(x.grad)
    def centerBack(self, x):
        z = 2*x
        y = x**2 - z + 1
        y.backward(retain_grad=True)
        print(z.grad)

if __name__=='__main__':
    simpleMaths()
