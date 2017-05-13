import data_helpers
import json
import rake
import w2v
from difflib import SequenceMatcher
from gensim.summarization import summarize
from nltk import word_tokenize, pos_tag
from keras.models import model_from_json

embedding_dim = 20
min_word_count = 1
context = 10

print('Loading Model...')
vocab = json.load(open('json/vocab.json'))
max_length = vocab["max_length"]
vocab.pop("max_length")
vocab_inv = json.load(open('json/vocab_inv.json'))
cat_tags = json.load(open('json/cat_tags.json'))
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
print('Model Loaded.')

content = open('data/computer.txt').read()

print('Extracting Keywords...')
rake_object = rake.Rake("data/Stoplist.txt", 2, 8, 1)
keys = rake_object.run(content)
keys = [i[0] for i in keys]
print('Extracted Keywords.')

print('Summarizing Content...')
content = summarize(content, 0.8, split=True)
print('Content Summarized.')

print('Linking Keywords to Sentences...')
keys_sentences = {x: list() for i, x in enumerate(keys)}

for key in keys:
    for sentence in content:
        if key in sentence:
            keys_sentences[key].append(sentence)
print('Linked Keywords to Sentences.')

with open('json/keys_sent.json', 'w') as out_file:
    json.dump(keys_sentences, out_file)

print("The document loaded is wikipedia computer article")
ques = input('Enter a question: ')
ques_keys = [j[0] for j in rake_object.run(ques)]
ques_X = data_helpers.get_question_vocab(ques, vocab, max_length)
category = categories[model.predict_classes(ques_X, verbose=0)[0]]
print(category)

query_keys = list()
for i in ques_keys:
    if len(i.split()) > 1:
        i = i.split()
        for j in i:
            query_keys.append(j)
    else:
        query_keys.append(i)

match_percent = 0
ideal_candidate = list()
answer = ''
cat_count = 0
possible_answers = list()
for i in cat_tags[category]:
    for j in query_keys:
        if j in keys_sentences:
            for k in keys_sentences[j]:
                if k.strip() not in possible_answers:
                    tagged = list(pos_tag(word_tokenize(k)))
                    tagged = [tok_tagged[1] for tok_tagged in tagged]
                    percent = 0
                    tag_percent = SequenceMatcher(None, i, tagged).ratio()
                    key_percent = SequenceMatcher(None, query_keys, k.split()).ratio()
                    if tag_percent > key_percent:
                        percent = tag_percent
                    else:
                        percent = key_percent
                    if percent > match_percent:
                        print(k)
                        match_percent = percent
                        ideal_candidate = tagged
                        answer = k
                        possible_answers.append(k.strip())
    cat_count += 1
print(answer)
