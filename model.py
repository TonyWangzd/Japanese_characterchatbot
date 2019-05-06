# coding:utf-8
import numpy as np
import sys
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
#EOS_id
EOS_ID = 2
# 神经元数
size = 8
#max input symbols
num_encoder_symbols = 12
#max output symbols
num_decoder_symbols = 18
#learninng rate
learning_rate = 0.1



def get_samples():

    ##构造奇数参数来训练seq2seq模型
    train_set = [[[5, 7, 9], [11, 13, 15, EOS_ID]], [[7, 9, 11], [13, 15, 17, EOS_ID]]]
    raw_encoder_input = []
    raw_decoder_input = []
    for sample in train_set:
        raw_encoder_input.append([PAD_ID] * (input_seq_len - len(sample[0])) + sample[0])
        raw_decoder_input.append([GO_ID] + sample[1] + [PAD_ID] * (output_seq_len - len(sample[1]) - 1))

    # therefore
    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    for length_idx in range(input_seq_len):
        encoder_inputs.append(np.array([encoder_input[length_idx] for encoder_input in raw_encoder_input], dtype=np.int32))

    for length_idx in range(output_seq_len):
        decoder_inputs.append(np.array([decoder_input[length_idx] for decoder_input in raw_decoder_input], dtype=np.int32))
        target_weights.append(np.array([

            0.0 if length_idx == output_seq_len - 1
                        or decoder_input[length_idx] == PAD_ID else 1.0
                        for decoder_input in raw_decoder_input
        ], dtype=np.float32))
    return encoder_inputs, decoder_inputs, target_weights

def get_model(feed_previous=False):
    encoder_inputs = []
    decoder_inputs = []
    target_weights = []

    for i in range(input_seq_len):
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                             name = "encoder{0}".format(i)))
    for i in range(output_seq_len + 1):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                              name = "decoder{0}".format(i)))
    for i in range(output_seq_len):
        target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                              name = "weight{0}".format(i)))

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
        feed_previous=feed_previous,
        dtype=tf.float32
    )
    #calculate the loss
    loss = seq2seq.sequence_loss(outputs, targets, target_weights)

    opt = tf.train.GradientDescentOptimizer(learning_rate)
    #optimize
    update = opt.apply_gradients(opt.compute_gradients(loss))
    #save model
    saver = tf.train.Saver(tf.global_variables())

    return encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, targets


def run_model():
    # training
    with tf.Session() as sess:
        sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = get_samples()
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, targets = get_model()

        input_feed = {}
        for l in range(input_seq_len):
            input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
        for l in range(output_seq_len):
            input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
            input_feed[target_weights[l].name] = sample_target_weights[l]

        input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)

        #outputs = sess.run(outputs,input_feed)
        #print(outputs)
        sess.run(tf.global_variables_initializer())

        for step in range(200):
            [loss_ret, _] = sess.run([loss, update], input_feed)
            if step % 10 == 0:
                print('step=', step, 'loss=', loss_ret)

        #save model
        saver.save(sess, './model/demo')


def predict():

    with tf.Session() as sess:
        sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = get_samples()
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, targets = get_model(feed_previous=True)
        # 从文件恢复模型
        saver.restore(sess, './model/demo')

        input_feed = {}
        for l in range(input_seq_len):
            input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
        for l in range(output_seq_len):
            input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
            input_feed[target_weights[l].name] = sample_target_weights[l]

        input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)

        # 预测输出
        outputs = sess.run(outputs, input_feed)

        for sample_index in range(2):
            # 因为输出数据每一个是num_decoder_symbols维的
            # 因此找到数值最大的那个就是预测的id，就是这里的argmax函数的功能
            outputs_seq = [int(np.argmax(logit[sample_index], axis=0)) for logit in outputs]
            # 如果是结尾符，那么后面的语句就不输出了
            if EOS_ID in outputs_seq:
                outputs_seq = outputs_seq[:outputs_seq.index(EOS_ID)]
            outputs_seq = [str(v) for v in outputs_seq]
            print (" ".join(outputs_seq))


if __name__ == "__main__":
    #if sys.argv[1] == 'train':
        #run_model()
    #else:
        predict()