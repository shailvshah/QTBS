from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np


def load_model(vocab_inv, num_features, min_word_count, context):
    model_dir = "word2vec_models"
    model_name = "{:d}features_{:d}minwords_{:d}context".format(
        num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print("Loading existing Word2Vec model '%s'" % split(model_name)[-1])
    embedding_weights = [np.array([embedding_model[w] if w in embedding_model
                                   else np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                                   for w in vocab_inv])]
    return embedding_weights

def train_word2vec(question_matrix, vocab_inv,
                   num_features=300, min_word_count=1, context=10):
    model_dir = "word2vec_models"
    model_name = "{:d}features_{:d}minwords_{:d}context".format(
        num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print("Loading existing Word2Vec model '%s'" % split(model_name)[-1])
    else:
        num_workers = 5
        downsampling = 1e-3
        print("Training Word2Vec model...")
        questions = [[vocab_inv[j] for j in i] for i in question_matrix]
        embedding_model = word2vec.Word2Vec(questions, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)
        embedding_model.init_sims(replace=True)
        if not exists(model_dir):
            os.mkdir(model_dir)
        print("Saving Word2Vec model '%s'" % split(model_name)[-1])
        embedding_model.save(model_name)

    embedding_weights = [np.array([embedding_model[w] if w in embedding_model
                                   else np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                                   for w in vocab_inv])]
    return embedding_weights

if __name__ == '__main__':
    import data_helpers
    print("Loading data...")
    x, _, _, vocabulary_inv = data_helpers.load_data()
    w = train_word2vec(x, vocabulary_inv)
    print(w)
