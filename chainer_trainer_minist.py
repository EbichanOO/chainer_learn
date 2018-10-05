from chainer.datasets import mnist
from chainer.backends import cuda
from chainer import training
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

def DataFormatChange(trainData, testData, batchsize=128):
    train_iter = iterators.SerialIterator(trainData, batchsize)
    test_iter = iterators.SerialIterator(testData, batchsize, False, False)
    return train_iter, test_iter

class MLP(Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

if __name__ == '__main__':
    train, test = mnist.get_mnist()
    train, test = DataFormatChange(train, test)

    gpu_id = -1 # set to 0 or more if use GPU
    model = MLP()
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    max_epock = 10
    # Wrap your model by Classifier and include the process of loss calculation within your model.
    # Since we do not specify a loss function here, the default 'softmax_cross_entropy' is used.
    model = L.Classifier(model)
    # selection of your optimizing method
    optimizer = optimizers.MomentumSGD()
    # Give the optimizer a reference to the model
    optimizer.setup(model)
    # Get an updater that uses the Iterator and Optimizer
    updater = training.updaters.StandardUpdater(train, optimizer, device=gpu_id)

    # Setup a Trainer
    trainer = training.Trainer(updater, (max_epock, 'epoch'), out='mnist_result')

    from chainer.training import extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(test, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()