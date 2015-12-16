import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from collections import OrderedDict
import numpy as np

import cPickle
import gzip

floatX = theano.config.floatX

<<<<<<< HEAD
train_set, test_set = cPickle.load(gzip.open('nao_densa_massa_560tr_500te_512px_original.pkl.gz', 'rb'))
=======
train_set, test_set = cPickle.load(gzip.open('nao_densa_massa_560tr_500te_768px_original.pkl.gz', 'rb'))
>>>>>>> d1ecd98649bbe1ab02899756f8788571f42da0a7

train_x,train_y = train_set
test_x,test_y = test_set

<<<<<<< HEAD
train_x = train_x.reshape(560, 262144)
train_y = train_y.reshape(560, 238144)

test_x = test_x.reshape(1, 262144)
test_y = test_y.reshape(1, 238144)
=======
train_x = train_x.reshape(560, 589824)
train_y = train_y.reshape(560, 553536)

test_x = test_x.reshape(1, 589824)
test_y = test_y.reshape(1, 553536)
>>>>>>> d1ecd98649bbe1ab02899756f8788571f42da0a7


def shared(X):
    return theano.shared(X.astype(floatX))


def batch_iterator(x, y, batch_size):
    num_batches = x.shape[0] // batch_size
    for i in xrange(num_batches):
        first = i * batch_size
        last  = (i+1) * batch_size
<<<<<<< HEAD
        x_batch = x[first:last].reshape((batch_size, 1, 512, 512))
        y_batch = y[first:last].reshape((batch_size, 1, 488, 488))
=======
        x_batch = x[first:last].reshape((batch_size, 1, 768, 768))
        y_batch = y[first:last].reshape((batch_size, 1, 744, 744))
>>>>>>> d1ecd98649bbe1ab02899756f8788571f42da0a7
        yield (x_batch, y_batch)


def convolutional_pooling_relu(input_channels, output_channels, filter_dim, X, Y):
    #n_in = input_channels * filter_dim * filter_dim
    #n_out = ((output_channels * filter_dim * filter_dim) / 4)
    #w_init = np.sqrt(6. / (n_in + n_out))
    #W = shared(np.random.uniform(low=-w_init, high=w_init, size=(output_channels, input_channels, filter_dim, filter_dim)))
    W = shared(np.random.normal(0, .001, size=(output_channels, input_channels, filter_dim, filter_dim)))
    b = shared(np.zeros((output_channels,)))
    Z = conv.conv2d(X, W) + b.dimshuffle('x', 0, 'x', 'x')
    DS = downsample.max_pool_2d(Z, ds=[2, 2])
    DS_gt = downsample.max_pool_2d(Y, ds=[2, 2])
    O = T.switch(DS > 0, DS, 0)
    #O_gt = T.switch(DS_gt > 0, DS_gt, 0)
    return W, b, O, DS_gt


def convolutional_relu(input_channels, output_channels, filter_dim, X, Y):
    #n_in = input_channels * filter_dim * filter_dim
    #n_out = ((output_channels * filter_dim * filter_dim) / 4)
    #w_init = np.sqrt(6. / (n_in + n_out))
    #W = shared(np.random.uniform(low=-w_init, high=w_init, size=(output_channels, input_channels, filter_dim, filter_dim)))
    W = shared(np.random.normal(0, .001, size=(output_channels, input_channels, filter_dim, filter_dim)))
    b = shared(np.zeros((output_channels,)))
    Z = conv.conv2d(X, W) + b.dimshuffle('x', 0, 'x', 'x')
    O = T.switch(Z > 0, Z, 0)
    #O_gt = T.switch(Y > 0, Y, 0)
    return W, b, O, Y


def convolutional_sig(input_channels, output_channels, filter_dim, X, Y):
    #n_in = input_channels * filter_dim * filter_dim
    #n_out = ((output_channels * filter_dim * filter_dim) / 4)
    #w_init = np.sqrt(6. / (n_in + n_out))
    #W = shared(np.random.uniform(low=-w_init, high=w_init, size=(output_channels, input_channels, filter_dim, filter_dim)))
    W = shared(np.random.normal(0, .001, size=(output_channels, input_channels, filter_dim, filter_dim)))
    b = shared(np.zeros((output_channels,)))
    Z = conv.conv2d(X, W) + b.dimshuffle('x', 0, 'x', 'x')
    O = T.nnet.sigmoid(Z)
    cost = T.nnet.binary_crossentropy(O, Y).mean()
    return W, b, O, cost

X = T.tensor4('X')
Y = T.tensor4('Y')

W0, b0, O0, GT0 = convolutional_pooling_relu(1, 16, 7, X, Y)
W1, b1, O1, GT1 = convolutional_pooling_relu(16, 32, 7, O0, GT0)
W2, b2, O2, GT2 = convolutional_relu(32, 16, 3, O1, GT1)
W3, b3, O3, cost = convolutional_sig(16, 1, 1, O2, GT2)

<<<<<<< HEAD
x = np.ones((1, 1, 512, 512)).astype(floatX)
y = np.ones((1, 1, 488, 488)).astype(floatX)
=======
x = np.ones((1, 1, 768, 768)).astype(floatX)
y = np.ones((1, 1, 744, 744)).astype(floatX)
>>>>>>> d1ecd98649bbe1ab02899756f8788571f42da0a7

FCNN = theano.function([X, Y], [O3, cost])
o, c = FCNN(x, y)

print c

params = [W0, b0, W1, b1, W2, b2, W3, b3]

updates = dict()

for p in params:
    updates[p] = p - .01 * T.grad(cost, p)
updates = OrderedDict(updates)

trainer = theano.function([X,Y], cost, updates=updates)

num_epochs = 2

for i in range(1, num_epochs):
    print('-'*10)
    print('Epoch: {}'.format(i))
    for iter, b in enumerate(batch_iterator(train_x, train_y, 1)):
        x = b[0]
        y = b[1]
        last_cost = trainer(x, y)

    print('cost: {}'.format(trainer(x, y)))


save_file_w0 = open('save_w0', 'wb')
save_file_w1 = open('save_w1', 'wb')
save_file_w2 = open('save_w2', 'wb')
save_file_w3 = open('save_w3', 'wb')

save_file_b0 = open('save_b0', 'wb')
save_file_b1 = open('save_b1', 'wb')
save_file_b2 = open('save_b2', 'wb')
save_file_b3 = open('save_b3', 'wb')

cPickle.dump(W0.get_value(borrow=True), save_file_w0, -1)
cPickle.dump(W1.get_value(borrow=True), save_file_w1, -1)
cPickle.dump(W2.get_value(borrow=True), save_file_w2, -1)
cPickle.dump(W3.get_value(borrow=True), save_file_w3, -1)

cPickle.dump(b0.get_value(borrow=True), save_file_b0, -1)
cPickle.dump(b1.get_value(borrow=True), save_file_b1, -1)
cPickle.dump(b2.get_value(borrow=True), save_file_b2, -1)
cPickle.dump(b3.get_value(borrow=True), save_file_b3, -1)

save_file_w0.close()
save_file_w1.close()
save_file_w2.close()
save_file_w3.close()

save_file_b0.close()
save_file_b1.close()
save_file_b2.close()
save_file_b3.close()
