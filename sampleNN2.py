import chainer
import chainer.functions as F
import chainer.links as L
#動きません

class SampleNN(chainer.Chain):
    def __init__(self, size=20, hidden=40):
        super(SampleNN, self).__init__()
        with self.init_scope():
            self.nn1 = L.Linear(None, hidden)
            self.nn2 = L.Linear(hidden, hidden)
            self.nn3 = L.Linear(hidden, size)
    def __call__(self, x):
        out = F.relu(self.nn1(x))
        out = F.relu(self.nn2(out))
        return self.nn3(out)

if __name__=='__main__':
    import numpy as np
    data_size = 20
    x = np.array([2 * np.pi * np.random.rand(20) for i in range(data_size)])
    y = np.sin(x)
    x = chainer.Variable(x.astype(np.float32))
    y = chainer.Variable(y.astype(np.float32))

    model = SampleNN()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    epoch = 10
    for i in range(epoch):
        for j in range(data_size):
            loss = F.mean_squared_error(model(x[j]), y[j])
            print("loss = {loss}")
            loss.backward()
            optimizer.update()

    print("--------test----------")
    
    x = 2*np.pi*np.random.rand(20)
    y = np.sin(x)
    loss = F.mean_squared_error(model(x), y)
    print("loss = {loss}")