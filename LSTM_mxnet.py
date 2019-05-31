import nltk
import numpy as np
from mxnet import nd
import random
import zipfile
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

file = './data/obama.txt'

num_steps = 3

def process_corpus(corpus_chars):
    corpus_chars = corpus_chars.replace('\n',' ').replace('\r',' ')
    corpus_words = nltk.word_tokenize(corpus_chars)

    index_to_word = list(set(corpus_words))
    word_to_index = dict([(word, i) for i, word in enumerate(index_to_word)])

    vocabulary_size = len(word_to_index)
    print('vocabulary_size:',vocabulary_size)

    corpus_indices = [word_to_index[word] for word in corpus_words]
    sample = corpus_chars[:40]
    print('sample:\n',sample)

def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices,ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len//batch_size

    indices = corpus_indices[0: batch_size*batch_len].reshape((batch_size,batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        data = indices[:,i:i+num_steps]
        label = indices[:,i+1:i+num_steps+1]
        yield data, label

def rnn(inputs, H, W_xh, W_hh, b_h, W_hy, b_y):
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh), b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H)





if __name__ == "__main__":
    text = open(file,encoding='utf=8').read()
    process_corpus(text)
