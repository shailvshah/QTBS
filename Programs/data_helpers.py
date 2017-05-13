import numpy as np
import re
import itertools
import json
from collections import Counter
from nltk import word_tokenize, pos_tag

lemma = lambda x: re.sub(r"[`?\n',]", '', x.lower().strip()).split()


def get_category_vector(categories, category):
    vector = np.zeros(len(categories), dtype='int32')
    for i in range(len(categories)):
        if categories[i] == category:
            vector[i] = 1
    return vector


def read_and_format_input_and_output_data():
    questions = list()
    categories = [i.replace('\n', '')
                  for i in list(open('data/categories.txt').readlines())]
    vectors = list()
    for i in open('data/questions.txt').readlines():
        i = i.split(' ', 1)
        i[0] = i[0].replace(':', '_')
        questions.append(lemma(i[1]))
        vectors.append(get_category_vector(categories, i[0]))
    labels = np.asarray(vectors)
    return [questions, labels, categories]


def pad_questions(questions, padding_word='<PAD/>'):
    max_length = max(len(x) for x in questions)
    padded_questions = list()
    for i in range(len(questions)):
        padded_question = questions[
            i] + [padding_word] * (max_length - len(questions[i]))
        padded_questions.append(padded_question)
    return [padded_questions, max_length]


def build_vocabulary(questions, max_length):
    word_counts = Counter(itertools.chain(*questions))
    idx_to_word = [x[0] for x in word_counts.most_common()]
    idx_to_word.append('<OTHER/>')
    word_to_idx = {x: i for i, x in enumerate(idx_to_word)}
    word_to_idx["max_length"] = max_length
    with open('json/vocab.json', 'w') as out:
        json.dump(word_to_idx, out)
    word_to_idx.pop("max_length")
    with open('json/vocab_inv.json', 'w') as out:
        json.dump(idx_to_word, out)
    return [word_to_idx, idx_to_word]


def build_input_data(questions, labels, vocab):
    X = np.array([[vocab[j] if j in vocab else vocab['<OTHER/>']
                   for j in i] for i in questions])
    Y = np.array(labels)
    return [X, Y]


def build_document_data(document, vocab):
    X = np.array([[vocab[j] for j in i] for i in document])
    return X


def get_question_vocab(question, vocab, max_length):
    question = lemma(question)
    question = question + ['<PAD/>'] * (max_length - len(question))
    X = np.array([[vocab[i] if i in vocab else vocab['<OTHER/>']
                   for i in question]])
    return X


def load_data():
    questions, labels, categories = read_and_format_input_and_output_data()
    questions_padded, max_length = pad_questions(questions)
    vocab, vocab_inv = build_vocabulary(questions_padded, max_length)
    X, Y = build_input_data(questions_padded, labels, vocab)
    return [X, Y, vocab, vocab_inv, max_length, categories]

def get_question_answers(qa):
    answers_list = list()
    for answer in qa['answers']:
        tagged = list(pos_tag(word_tokenize(answer['text'])))
        answers_list.append([tok_tagged[1] for tok_tagged in tagged])
    return answers_list

def read_test_data():
    questions = list()
    answers = list()
    question_json = json.load(open('data/dev-v1.1.json'))
    for document in question_json['data']:
        for paragraph in document['paragraphs']:
            for qas in paragraph['qas']:
                i = qas['question']
                questions.append(i)
                answers.append(get_question_answers(qas))
    return [questions, answers]

def pad_test_data(questions, max_length, padding_word='<PAD/>'):
    padded_questions = list()
    for i in range(len(questions)):
        padded_question = questions[i] + [padding_word] * (max_length - len(questions[i]))
        padded_questions.append(padded_question)
    return padded_questions

def load_test_data(max_length, vocab):
    questions = read_test_data()

