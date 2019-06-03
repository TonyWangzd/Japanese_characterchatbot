import nltk
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time
from sklearn.externals import joblib
import numpy as np
import os


file = './data/obama2.txt'

num_steps = 10

num_epochs = 1000

clipping_theta = 0.01

lr = 100

batch_size = 10

pred_period = 1

pred_len = 30

para_file = 'params_rnn.params'



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

def sample(preds, temprature):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temprature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

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
    H = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, H


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
                num_hiddens, vocab_size, idx_to_char, char_to_idx, batch_size):
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

def grad_clipping(params, theta):
    norm = nd.array([0])
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):

    data_iter_fn = data_iter_consecutive
    if os.path.exists(para_file):
        params = nd.load(para_file)
        for param in params:
            param.attach_grad()
        print('successfully load the params before...\n')
    else:
        params = get_params()
        print('generated new params to training')

    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if (epoch % 10) == 0:
            lr = 0.8*lr
        print('training...now is on epoch'+str(epoch)+'\n')
        #在epoch开始时初始化隐藏状态
        state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            for s in state:
                s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)
                # 拼接之后形状为(num_steps * batch_size, vocab_size)
                outputs = nd.concat(*outputs, dim=0)
                # Y的形状是(batch_size, num_steps)，转置后再变成长度为
                # batch * num_steps 的向量，这样跟输出的行一一对应
                y = Y.T.reshape((-1,))
                # 使用交叉熵损失计算平均分类误差
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta)  # 裁剪梯度
            sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.asscalar() * y.size
            n += y.size
        #print loss
        print('loss now is' + str(l_sum))

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -',
                    predict_rnn(
                    prefix, pred_len, rnn, params, num_hiddens, vocab_size, idx_to_char, char_to_idx, batch_size))
            # 保存x
            nd.save(para_file, params)
            print('save success to params_rnn.para')

if __name__ == "__main__":

    ctx = None

    text = open(file, encoding='utf=8').read()

    index_to_word, word_to_index, corpus_indices, vocabulary_size = process_corpus(text)

    num_inputs, num_hiddens, num_outputs = vocabulary_size, 256, vocabulary_size
    # parameter for three layers
    params = get_params()
    prefixes = [['who', 'is'],['I','am']]
    #result = predict_rnn(prefix, 10, rnn, params, num_hiddens, vocabulary_size, index_to_word, word_to_index)
    #print(result)
    train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocabulary_size, ctx, corpus_indices, index_to_word,
                              word_to_index, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)



