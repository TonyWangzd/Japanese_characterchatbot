import numpy as np
from keras import layers
import pandas as pd
import keras
import random
import sys

maxlen = 60

step = 3

sentences = []

next_chars = []


def reweight_distribution(original_distribution, tempreture=0.5):
    distribution = np.log(original_distribution) / tempreture
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)


def sample(preds, temprature=0.5):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temprature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def get_model():
    path = keras.utils.get_file(
        'nietzsche.txt',
        origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read().lower()
    print('Corpus length', len(text))

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('Number of sequences:', len(sentences))

    # pd_sentences = pd.Series(sentences)

    chars = sorted(list(set(text)))
    print('Unique characters:', len(chars))
    char_indices = dict((char, chars.index(char)) for char in chars)

    # one hot encoding
    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    model = keras.models.Sequential()
    model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(layers.Dense(len(chars), activation='softmax'))
    model.summary()

    optimizer = keras.optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    for epoch in range(1, 60):
        print('epoch', epoch)
        model.fit(x, y, batch_size=128,
                  epochs=2)
        start_index = random.randint(0,len(text) - maxlen - 1)
        generated_text = text[start_index: start_index + maxlen]
        print('-----generating with seed: "'+ generated_text + '"')

        for temprature in [0.2, 0.5, 1.0, 1.2]:
            print('------temperature:', temprature)
            sys.stdout.write(generated_text)

            # generate 400 characters
            for i in range(400):
                sampled = np.zeros((1, maxlen, len(chars)))
                for t,char in enumerate(generated_text):
                    sampled[0, t, char_indices[char]] = 1

                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temprature)
                next_char = chars[next_index]

                generated_text += next_char
                generated_text = generated_text[1:]

                sys.stdout.write(next_char)
                sys.stdout.flush()
        print()

if __name__ == "__main__":
    get_model()



