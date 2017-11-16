import string
import json
import numpy as np
import lda
from nltk.corpus import stopwords
import enchant

stop = stopwords.words("english")
def valid(word, stop):
    word = word.lower()
    if word in stop or len(word) <= 2:
        return False
    return True

translator = str.maketrans('', '', string.punctuation)
vocab = {}
descriptions = []
titles = []
for i in range(1,51):
    with open('../upwork-learning/freelancer-com-scrape/data' + str(i) + '.json') as jf:
        jd = json.load(jf)
        for d in jd:
            cd = d['desc'].translate(translator)
            
            # save description
            descriptions.append(cd)
            
            titles.append(d['title'])

            for w in cd.split():
                w = w.lower()
                if not valid(w, stop):
                    continue
                if w not in vocab.keys():
                    vocab[w] = 1

vi = list(vocab.keys())

print("total:", str(len(descriptions)))

mat = None
cnt = 1
for desc in descriptions:
    if cnt % 200 == 0:
        print("count:", cnt)
    cnt += 1
    row = np.zeros((1, len(vi)), dtype=np.int)
    for w in desc.split():
        w = w.lower()
        if not valid(w, stop):
            continue
        row[0][vi.index(w)] += 1
    if mat is None:
        mat = row
    else:
        mat = np.concatenate((mat, row), axis=0)

print(mat.shape)
print(mat)

model = lda.LDA(n_topics=7, n_iter=1500, random_state=1)
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
    #print('Topic {}: {}'.format(i, ' '.join(topic_words)))

with open("topic-words.json", "w") as topicout:
    json.dump(topic_words_obj, topicout)

doc_topic = model.doc_topic_
doc_topics_obj = {}
for i in range(len(doc_topic)):
    doc_topics_obj[str(i)] = {'topic': str(doc_topic[i].argmax()), 'title': str(titles[i]), 'desc': str(descriptions[i])}
    #print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))

with open("doc-topics.json", "w") as docout:
    json.dump(doc_topics_obj, docout)
