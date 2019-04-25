import tensorflow as tf
import nltk
import gensim

def rnn_model(model = 'lstm', input_data, output_data, vocab_size,
              run_size = 128, num_layers = 2, batch_size = 64, learning_rate = 0.001):
    end_points = {}
    cell_fun = tf.contrib.rnn.BasicLSTMCell
    cell = cell_fun(run_size)
    cell = tf.contrib.rnn.MultiRNNCell([cell]*num_layers)
    initial_state = cell.zero_state(batch_size,tf.float32)
    embedding = tf.Variable(tf.random_uniform([vocab_size + 1,run_size],-1.0,1.0))
    inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = tf.nn.(cell,inputs,initial_state = initial_state)
    output = tf.reshape(outputs, [-1,run_size])
    weights = tf.Variable(tf.truncated_normal([run_size,vocab_size+1]))
    bias = tf.Variable(tf.zeros(shape = vocab_size+1))
    logits = tf.nn.bias_add(tf.matmul(output,weights),bias = bias)

    labels = tf.one_hot(tf.reshape(output_data,[-1]),depth=vocab_size+1)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits=logits)
    total_loss = tf.reduce_mean(loss)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    end_points['initial_state'] = initial_state
    end_points['output'] = output
    end_points['train_op'] = train_op
    end_points['total_loss'] = total_loss






