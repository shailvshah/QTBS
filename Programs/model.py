import numpy as np
import json
import time
from os.path import exists
# from squad import data_helpers, w2v
import data_helpers
import w2v
from keras.models import Sequential, Model, model_from_json
from keras.layers import Activation, Dense, Dropout, Embedding
from keras.layers import Flatten, Input, Merge, Convolution1D, MaxPooling1D

np.random.seed(2)

embedding_dim = 300
filter_sizes = (2, 3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150
batch_size = 128
num_epochs = 1
val_split = 0.3
min_word_count = 1
context = 3

# if exists('model.json') and exists('model.h5'):
#     vocab = json.load(open('json/vocab.json'))
#     max_length = vocab["max_length"]
#     vocab.pop("max_length")
#     vocab_inv = json.load(open('json/vocab_inv.json'))
#     unique_answers = json.load(open('json/answers.json'))
#     embedding_weights = w2v.load_model(vocab_inv, embedding_dim, min_word_count, context)
#     print('Loading model from disk...')
#     json_file = open('model.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     model = model_from_json(loaded_model_json)
#     # load weights into new model
#     model.load_weights('model.h5')
#     print('Loaded model from disk')
#     model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

if True:
    print('Loading Data...')
    start = time.time()
    X, Y, vocab, vocab_inv, max_length, unique_answers = data_helpers.load_data()
    end = time.time()
    print('Loaded data successfully took ', (end - start), ' seconds')
    embedding_weights = w2v.train_word2vec(X, vocab_inv, embedding_dim, min_word_count, context)
    # vocab = json.load(open('json/vocab.json'))
    # vocab_inv = json.load(open('json/vocab_inv.json'))
    # unique_answers = json.load(open('json/answers.json'))
    # max_length = vocab["max_length"]
    # vocab.pop("max_length")
    # embedding_weights = w2v.load_model(vocab_inv, embedding_dim, min_word_count, context)
    graph_in = Input(shape=(max_length, embedding_dim))
    convolutions = []
    for fsz in filter_sizes:
        convolution = Convolution1D(nb_filter=num_filters,
                                    filter_length=fsz,
                                    border_mode='valid',
                                    activation='relu',
                                    subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(convolution)
        flatten = Flatten()(pool)
        convolutions.append(flatten)

    if len(filter_sizes) > 1:
        out = Merge(mode='concat')(convolutions)
    else:
        out = convolutions[0]

    graph = Model(input=graph_in, output=out)

    model = Sequential()
    model.add(Embedding(len(vocab), embedding_dim, input_length=max_length, weights=embedding_weights))
    model.add(Dropout(dropout_prob[0], input_shape=(max_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(len(unique_answers)))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X, Y, batch_size=batch_size, nb_epoch=num_epochs, validation_split=val_split)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")

question = input('Enter a question: ')
question = data_helpers.get_question_input(question, max_length, vocab)
print(question)
print(unique_answers[model.predict_classes(question)[0]])

question = input('Enter a question: ')
question = data_helpers.get_question_input(question, max_length, vocab)
print(question)
print(unique_answers[model.predict_classes(question)[0]])

