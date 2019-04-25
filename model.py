# coding:utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

# 输入序列长度
input_seq_len = 5
# 输出序列长度
output_seq_len = 5
# 空值填充0
PAD_ID = 0
# 输出序列起始标记
GO_ID = 1

def get_samples():

    ##构造奇数参数来训练seq2seq模型
    train_set = [[[1, 3, 5], [7, 9, 11]], [[3, 5, 7], [9, 11, 13]]]

    # 所以这样一来第一个样本输出为

    encoder_input_0 = [PAD_ID] * (input_seq_len - len(train_set[0][0])) + train_set[0][0]

    # 同理第二个输出为
    encoder_input_1 = [PAD_ID] * (input_seq_len - len(train_set[1][0])) + train_set[1][0]

    # decoder_input 使用go_id作为起始
    decoder_input_0 = [GO_ID] + train_set[0][1] + [PAD_ID] * (output_seq_len - len(train_set[0][1]) - 1)
    decoder_input_1 = [GO_ID] + train_set[1][1] + [PAD_ID] * (output_seq_len - len(train_set[1][1]) - 1)

    # therefore
    encoder_inputs = []
    decoder_inputs = []

    for length_idx in range(input_seq_len):
        encoder_inputs.append(np.array([encoder_input_0[length_idx], encoder_input_1[length_idx]], dtype=np.int32))

    for length_idx in range(output_seq_len):
        decoder_inputs.append(np.array([decoder_input_0[length_idx], decoder_input_1[length_idx]], dtype=np.int32))

    return encoder_inputs, decoder_inputs

def get_model():
    encoder_inputs = []
    decoder_inputs = []

    for i in range(input_seq_len):
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                             name = 'encoder{0}'.format(i)))
    for i in range(output_seq_len):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                              name = 'decoder{0}'.format(i)))

    size = 8

    cell_fun = tf.contrib.rnn.BasicLSTMCell
    cell = cell_fun(size)
    num_encoder_symbols = 10
    num_decoder_symbols = 16

    outputs, _ = seq2seq.embedding_attention_seq2seq(
        encoder_inputs,
        decoder_inputs[:output_seq_len],
        cell,
        num_encoder_symbols = num_encoder_symbols,
        num_decoder_symbols = num_decoder_symbols,
        embedding_size = size,
        output_projection=None,
        feed_previous=False,
        dtype=tf.float32
    )
    return encoder_inputs, decoder_inputs, outputs


def run_model():
    with tf.Session() as sess:
        sample_encoder_inputs, sample_decoder_inputs = get_samples()
        encoder_inputs, decoder_inputs, outputs = get_model()
        input_feed = {}
        for l in range(input_seq_len):
            input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
        for l in range(output_seq_len):
            input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]

        sess.run(tf.global_variables_initializer())
        outputs = sess.run(outputs,input_feed)
        print(outputs)


if __name__ == "__main__":
    run_model()