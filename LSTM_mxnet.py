import nltk
import numpy as np
from mxnet import nd
import random
import zipfile
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

file = './data/obama2.txt'

num_steps = 3

def process_corpus(corpus_chars):
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_words = nltk.word_tokenize(corpus_chars)

    index_to_word = list(set(corpus_words))
    word_to_index = dict([(word, i) for i, word in enumerate(index_to_word)])

    vocabulary_size = len(word_to_index)
    print('vocabulary_size:', vocabulary_size)

    corpus_indices = [word_to_index[word] for word in corpus_words]
    sample = corpus_chars[:40]
    print('sample:\n', sample)
    return index_to_word, word_to_index, corpus_indices, vocabulary_size

# 相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size

    indices = corpus_indices[0: batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        data = indices[:, i:i + num_steps]
        label = indices[:, i + 1:i + num_steps + 1]
        yield data, label
# 随机采样
#def data_iter_random(corpus_indices, batch_size, numsteps, ctx=None):


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


def to_onehot(X, size):
    onehot_vector = [nd.one_hot(x, size) for x in X]
    return onehot_vector


def init_rnn_state(batch_size, num_hiddens, ctx=None):
    rnn_state = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
    return rnn_state


def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=None)

    # hidden layer
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=None)
    # output layer
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=None)
    # 梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params


def predict_rnn(prefix, num_chars, rnn, params,
                num_hiddens, vocab_size, idx_to_char, char_to_idx, ctx=None):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        (Y, state) = rnn(X, state, params)

        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ' '.join([idx_to_char[i] for i in output])


if __name__ == "__main__":
    ctx = None
    text = open(file, encoding='utf=8').read()
    index_to_word, word_to_index, corpus_indices, vocabulary_size = process_corpus(text)
    #my_seq = list(range(30))
    #for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    #   print('X:', X, '\nY:', Y, '\n')
    #X = nd.arange(10).reshape((2, 5))
    #inputs = to_onehot(X, vocabulary_size)
    #print(len(inputs), inputs[0].shape)

    num_inputs, num_hiddens, num_outputs = vocabulary_size, 256, vocabulary_size
    # parameter for three layers
    params = get_params()
    prefix = ['who', 'is']
    result = predict_rnn(prefix, 10, rnn, params, num_hiddens, vocabulary_size, index_to_word, word_to_index)
    print(result)



