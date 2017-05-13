import itertools
import json
import numpy as np
import re
from collections import Counter
from nltk import word_tokenize, pos_tag, map_tag

lemma = lambda x: re.sub(r"[`?\n',]", '', x.strip())


def get_question_answers(qa):
    return [answer['text'] for answer in qa['answers']]


def read_and_process_data():
    counter = 0
    question_json = json.load(open('json/dev-v1.1.json'))
    questions = list()
    answers = list()
    for document in question_json['data']:
        for paragraph in document['paragraphs']:
            for qas in paragraph['qas']:
                if counter > 5000:
                    return [questions, answers]
                questions.append(qas['question'])
                answers.append(get_question_answers(qas))
                counter += 1
    return [questions, answers]


def tag_answers(answers, all_answers_tagged):
    answers_tagged = list()
    for answer in answers:
        tagged = list(pos_tag(word_tokenize(lemma(answer))))
        all_answers_tagged.append(','.join([tok_tagged[1] for tok_tagged in tagged]))
        answers_tagged.append(','.join([tok_tagged[1] for tok_tagged in tagged]))
    return answers_tagged


def tag_question_answers(questions, answers):
    questions_tagged = list()
    answers_tagged = list()
    all_answers_tagged = list()
    for i in range(len(questions)):
        tagged = list(pos_tag(word_tokenize(lemma(questions[i]))))
        tagged = ",".join([tok_tagged[1] if tok_tagged[1] == 'NNP' else tok_tagged[0] for tok_tagged in tagged])
        if tagged in questions_tagged:
            questions_tagged.append(tagged)
            answers_tagged.append(tag_answers(answers[i], all_answers_tagged))
    return [questions_tagged, answers_tagged, list(set(all_answers_tagged)), list(set(questions_tagged))]


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


def get_output_vector(unique_answers, answers):
    vector = np.zeros(len(unique_answers), dtype='int32')
    i = unique_answers.index(answers[0])
    vector[i] = 1
    return vector


def build_input_and_output_data(questions, answers, unique_answers, vocab):
    X_input = list()
    Y_input = list()
    for i in range(len(questions)):
        X_input.append([vocab[j] if j in vocab else vocab['<OTHER/>'] for j in questions[i]])
        Y_input.append(get_output_vector(unique_answers, answers[i]))
    X = np.array(X_input)
    Y = np.array(Y_input)
    return [X, Y]


def load_data():
    questions, answers = read_and_process_data()
    questions_tagged, answers_tagged, unique_answers, unique_questions = tag_question_answers(questions, answers)
    print(len(unique_questions))
    with open('json/answers.json', 'w') as out:
        json.dump(unique_answers, out)
    questions_padded, max_length = pad_questions(questions_tagged)
    vocabulary, vocabulary_inverse = build_vocabulary(questions_padded, max_length)
    X, Y = build_input_and_output_data(questions_padded, answers_tagged, unique_answers, vocabulary)
    print(len(X), len(Y))
    return [X, Y, vocabulary, vocabulary_inverse, max_length, unique_answers]


def get_question_input(question, max_length, vocab):
    tagged = list(pos_tag(word_tokenize(lemma(question))))
    question = [tok_tagged[1] if tok_tagged[1] == 'NNP' else tok_tagged[0] for tok_tagged in tagged]
    question = question + ['<PAD/>'] * (max_length - len(question))
    X = np.array([[vocab[i] if i in vocab else vocab['<OTHER/>'] for i in question]])
    return X


if __name__ == '__main__':
    load_data()
