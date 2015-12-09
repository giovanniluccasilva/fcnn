import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import numpy as np
import cv2 as cv
import cPickle

floatX = theano.config.floatX


def normalize_image(img):

    normalized_img = img

    min_v = normalized_img.min()
    max_v = normalized_img.max()

    height, width = normalized_img.shape

    for i in xrange(height):
        for j in xrange(width):
            normalized_img[i, j] = (img[i, j] - min_v) / (max_v - min_v)
            normalized_img[i, j] *= 255.0

    return normalized_img


def binarization(img):

    binarized_img = img

    height, width = binarized_img.shape

    for i in xrange(height):
        for j in xrange(width):
            if img[i, j] >= 230.0:
                binarized_img[i, j] = 255
            else:
                binarized_img[i, j] = 0

    return binarized_img


def convolutional_pooling_relu(X, W, b):
    Z = conv.conv2d(X, W) + b.dimshuffle('x', 0, 'x', 'x')
    DS = downsample.max_pool_2d(Z, ds=[2, 2])
    O = T.switch(DS > 0, DS, 0)
    return O


def convolutional_relu(X, W, b):
    Z = conv.conv2d(X, W) + b.dimshuffle('x', 0, 'x', 'x')
    O = T.switch(Z > 0, Z, 0)
    return O


def convolutional_sig(X, W, b):
    Z = conv.conv2d(X, W) + b.dimshuffle('x', 0, 'x', 'x')
    O = T.nnet.hard_sigmoid(Z)
    return O


X = T.tensor4('X')

w0 = cPickle.load(open('Model8/save_w0', 'rb'))
w1 = cPickle.load(open('Model8/save_w1', 'rb'))
w2 = cPickle.load(open('Model8/save_w2', 'rb'))
w3 = cPickle.load(open('Model8/save_w3', 'rb'))

b0 = cPickle.load(open('Model8/save_b0', 'rb'))
b1 = cPickle.load(open('Model8/save_b1', 'rb'))
b2 = cPickle.load(open('Model8/save_b2', 'rb'))
b3 = cPickle.load(open('Model8/save_b3', 'rb'))

W0 = theano.shared(w0)
W1 = theano.shared(w1)
W2 = theano.shared(w2)
W3 = theano.shared(w3)

b0 = theano.shared(b0)
b1 = theano.shared(b1)
b2 = theano.shared(b2)
b3 = theano.shared(b3)

O0 = convolutional_pooling_relu(X, W0, b0)
O1 = convolutional_pooling_relu(O0, W1, b1)
O2 = convolutional_relu(O1, W2, b2)
O3 = convolutional_sig(O2, W3, b3)


for i in range(500, 1000):
    img = cv.imread('/media/giovanni/Dados/DDSM_3/Processadas/Nao_Densa_Massa/Reduzida3/'+ str(i+1) +'.png', 0)
    x = np.asarray(img, dtype=np.float32)
    h, w = img.shape
    x = img.reshape(1, 1, h, w)

    FCNN = theano.function([X], [O3])
    o = FCNN(x)

    out = normalize_image(o[0][0, 0, :, :])
    image = binarization(out)

    cv.imwrite('/media/giovanni/Dados/DDSM_3/Processadas/Nao_Densa_Massa/Results/Model8/'+ str(i+1) +'.png', image)
