import numpy as np
import data_helpers
import json
import w2v
from os.path import exists
from keras.models import Sequential, Model, model_from_json
from keras.layers import Activation, Dense, Dropout, Embedding
from keras.layers import Flatten, Input, Merge, Convolution1D, MaxPooling1D

np.random.seed(2)

embedding_dim = 20
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150
batch_size = 128
num_epochs = 50
val_split = 0.1
min_word_count = 1
context = 10

if exists("neural_model/model.json") and exists("neural_model/model.h5"):
    vocab = json.load(open('json/vocab.json'))
    max_length = vocab["max_length"]
    vocab.pop("max_length")
    vocab_inv = json.load(open('json/vocab_inv.json'))
    categories = [i.replace('\n', '')
                  for i in list(open('data/categories.txt').readlines())]
    embedding_weights = w2v.load_model(vocab_inv, embedding_dim, min_word_count, context)
    print('Loading model from disk...')
    json_file = open('neural_model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('neural_model/model.h5')
    print('Loaded model from disk')
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
else:
    print("Loading Data...")
    X, Y, vocab, vocab_inv, max_length, categories = data_helpers.load_data()
    print("Loaded Data Successfully...")
    embedding_weights = w2v.train_word2vec(
        X, vocab_inv, embedding_dim, min_word_count, context)
    graph_in = Input(shape=(max_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    if len(filter_sizes) > 1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)
    model = Sequential()
    model.add(Embedding(len(vocab), embedding_dim, input_length=max_length,
                        weights=embedding_weights))
    model.add(Dropout(dropout_prob[0], input_shape=(
        max_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(len(categories)))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X, Y, batch_size=batch_size,
              nb_epoch=num_epochs, validation_split=val_split)
    model_json = model.to_json()
    with open("neural_model/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("neural_model/model.h5")
    print("Saved model to disk")

print('Loading Test Data...')
test_questions, test_answers = data_helpers.read_test_data()
print('Loaded Test Data.')

pos_tags = {x: list() for i, x in enumerate(categories)}

print('Predicting for ', len(test_questions))
for i in range(len(test_questions)):
    ques = data_helpers.get_question_vocab(test_questions[i], vocab, max_length)
    category = categories[model.predict_classes(ques, verbose=0)[0]]
    for j in test_answers[i]:
        pos_tags[category].append(j)

with open('json/cat_tags.json', 'w') as out:
    json.dump(pos_tags, out)
print('Predicted for the questions')
