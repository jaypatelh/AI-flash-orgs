import string
import json
import numpy as np
import lda
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
from nltk import ngrams
import enchant
import sys
import operator

stop = stopwords.words("english")
stemmer = SnowballStemmer("english")
def clean(word):
    word = word.lower()
    if word in stop or len(word) <= 2:
        return ""
    word = stemmer.stem(word)
    return word

def bigrams(desc):
    cleaned = []
    for w in desc.split():
        w = clean(w)
        if w == "":
            continue
        cleaned.append(w)
    
    bgms = ngrams(cleaned, 2)

    vocab = {}
    split_cd = []
    for b in bgms:
        s = b[0] + ' ' + b[1]
        if s not in vocab:
            vocab[s] = 1
        else:
            vocab[s] += 1
        split_cd.append(s)
    
    return vocab, split_cd

def unigrams(desc):
    vocab = {}
    split_cd = []
    for w in desc.split():
        w = clean(w)
        if w == "":
            continue
        if w not in vocab:
            vocab[w] = 1
        else:
            vocab[w] += 1
        split_cd.append(w)
    return vocab, split_cd

def add_dict_counts(dictMain, dictOther):
    for k in dictOther.keys():
        if k in dictMain:
            dictMain[k] += dictOther[k]
        else:
            dictMain[k] = dictOther[k]
    return dictMain

def filter_below_threshold(dct):
    final = {}
    all_counts = {}
    for k in dct.keys():
        # track frequency of counts
        st = str(dct[k])
        if st not in all_counts:
            all_counts[st] = 1
        else:
            all_counts[st] += 1
        
        # filter
        if dct[k] > 10:
            final[k] = dct[k]

    return final, all_counts

translator = str.maketrans('', '', string.punctuation)
vocab = {}
descriptions = []
titles = []
for i in range(1,51):
    with open('../freelancer-com-scrape/data' + str(i) + '.json') as jf:
        jd = json.load(jf)
        for d in jd:
            #combined = d['desc'] + ' ' + d['title']
            combined = d['desc']
            cd = combined.replace("[url removed, login to view]", "")
            cd = cd.translate(translator)

            vcb, split_cd = unigrams(cd)
            add_dict_counts(vocab, vcb)
            descriptions.append(split_cd)
            
            titles.append(d['title'])

vocab, all_counts = filter_below_threshold(vocab)
#for k in sorted(all_counts.keys(), key=lambda x: int(x)):
#    print(k, ' ', all_counts[k])
vi = list(vocab.keys())
vocab_size = len(vi)

print("total descriptions:", str(len(descriptions)))
print("vocab size:", str(vocab_size))

mat = None
cnt = 1
for desc in descriptions:
    if cnt > 5000:
        break
    if cnt % 200 == 0:
        print("count:", cnt)
    cnt += 1
    row = np.zeros((1, vocab_size), dtype=np.int)
    for w in desc:
        if w not in vocab:
            continue
        row[0][vi.index(w)] += 1
    if mat is None:
        mat = row
    else:
        mat = np.concatenate((mat, row), axis=0)

print(mat.shape)
print(mat)

model = lda.LDA(n_topics=4, n_iter=20000, random_state=1)
model.fit(mat)  # model.fit_transform(X) is also available

topic_word = model.topic_word_  # model.components_ also works
n_top_words = 5
topic_words_obj = {}
for i, topic_dist in enumerate(topic_word):
    sub = []
    for elem in np.argsort(topic_dist).astype(int):
        sub.append(vi[elem])
    topic_words = sub[:-n_top_words:-1]
    topic_words_obj[str(i)] = topic_words
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

"""
with open("topic-words.json", "w") as topicout:
    json.dump(topic_words_obj, topicout)
"""
doc_topic = model.doc_topic_
doc_topics_obj = {}
for i in range(len(doc_topic)):
    doc_topics_obj[str(i)] = {'topic': str(doc_topic[i].argmax()), 'title': str(titles[i]), 'desc': str(descriptions[i])}
    #print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
"""
with open("doc-topics.json", "w") as docout:
    json.dump(doc_topics_obj, docout)
"""