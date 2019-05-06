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
# 神经元数
size = 8
#max input symbols
num_encoder_symbols = 10
#max output symbols
num_decoder_symbols = 16


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
    target_weights = []
    for length_idx in range(input_seq_len):
        encoder_inputs.append(np.array([encoder_input_0[length_idx], encoder_input_1[length_idx]], dtype=np.int32))

    for length_idx in range(output_seq_len):
        decoder_inputs.append(np.array([decoder_input_0[length_idx], decoder_input_1[length_idx]], dtype=np.int32))
        target_weights.append(np.array([

            0.0 if length_idx == output_seq_len - 1
                        or decoder_input_0[length_idx] == PAD_ID else 1.0,

            0.0 if length_idx == output_seq_len - 1
                   or decoder_input_1[length_idx] == PAD_ID else 1.0,
        ], dtype=np.float32))
    return encoder_inputs, decoder_inputs, target_weights

def get_model():
    encoder_inputs = []
    decoder_inputs = []
    target_weights = []

    for i in range(input_seq_len):
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                             name = 'encoder{0}'.format(i)))
    for i in range(output_seq_len + 1):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                              name = 'decoder{0}'.format(i)))
    for i in range(output_seq_len):
        target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                              name = 'weight{0}'.format(i)))

    targets = [decoder_inputs[i + 1] for i in range(output_seq_len)]
    cell_fun = tf.contrib.rnn.BasicLSTMCell
    cell = cell_fun(size)

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
    #calculate the loss
    loss = seq2seq.sequence_loss(outputs, targets, target_weights)

    return encoder_inputs, decoder_inputs, target_weights, outputs, loss


def run_model():
    with tf.Session() as sess:
        sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = get_samples()
        encoder_inputs, decoder_inputs, target_weights, outputs, loss = get_model()

        input_feed = {}
        for l in range(input_seq_len):
            input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
        for l in range(output_seq_len):
            input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
            input_feed[target_weights[l].name] = sample_target_weights[l]

        input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)

        sess.run(tf.global_variables_initializer())
        #outputs = sess.run(outputs,input_feed)
        #print(outputs)
        sess.run(tf.global_variables_initializer())
        loss = sess.run(loss, input_feed)
        print(loss)

if __name__ == "__main__":
    run_model()