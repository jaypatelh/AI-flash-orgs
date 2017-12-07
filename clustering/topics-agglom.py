import string
import json
import numpy as np
import lda
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import AgglomerativeClustering
from nltk import ngrams
import enchant
import sys
import operator
import matplotlib.pyplot as plt
import sys

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
for i in range(1,140):
    with open('../mobile-phone/data' + str(i) + '.json') as jf:
        jd = json.load(jf)
        for d in jd:
            #combined = d['desc'] + ' ' + d['title']
            combined = d['desc']
            cd = combined.replace("[url removed, login to view]", "")
            sentences = cd.split(".")

            for sentence in sentences:
                cd = sentence.translate(translator)
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

cache = {}

def LCS(descA, descB):
  if str((descA, descB)) in cache:
    return cache[str((descA, descB))]
  if str((descB, descA)) in cache:
    return cache[str((descB, descA))]
  if len(descA)==0 or len(descB)==0:
    cache[str((descA, descB))] = []
    return []
  if descA[-1] == descB[-1]:
    comp = LCS(descA[:-1], descB[:-1])
    comp.append(descA[-1])
    cache[str((descA, descB))] = comp
    return comp
  else:
    candidate1 = LCS(descA[:-1], descB)
    candidate2 = LCS(descA, descB[:-1])
    if len(candidate1) >= len(candidate2):
      cache[str((descA, descB))] = candidate1
      return candidate1
    else:
      cache[str((descA, descB))] = candidate2
      return candidate2

def rouge_distance(descRef, descSys):
    if len(descRef) == 0 or len(descSys) == 0:
        return 1
    if descRef == descSys:
        return 0
    intersect = LCS(descRef, descSys)
    num_intersect = len(intersect)
    #print("num intersect: ", num_intersect)
    precision = float(num_intersect) / len(descSys)
    recall = float(num_intersect) / len(descRef)
    #print("precision: ", precision)
    #print("recall: ", recall)
    if precision + recall > 0:
        f1 = 2 * (float(precision * recall) / (precision + recall))
        #print("f1: ", f1)
        f1_dist = np.exp(-1 * f1)
        #print("f1 dist: ", f1_dist)
    else:
        f1_dist = 1
    return f1_dist

"""
print(rouge_distance(['hello how', 'how are', 'are you'], ['hello how', 'how are', 'are you']))
print("=================")
print(rouge_distance(['hello how', 'how are', 'are you'], ['hello how', 'how is', 'is you']))
print("=================")
print(rouge_distance(['hello how', 'how are', 'are you'], ['blah how', 'how is', 'is you']))
"""

first_few = descriptions[:600]
mat = None
for idx, desc in enumerate(first_few):
    print("on ", idx)
    row = np.zeros((1, 1000), dtype=np.float)
    for idx2, desc2 in enumerate(first_few):
        row[0][idx2] = rouge_distance(desc, desc2)
    if mat is None:
        mat = row
    else:
        mat = np.concatenate((mat, row), axis=0)

print(mat.shape)
print(mat)

model = AgglomerativeClustering(n_clusters=35, )
model.fit(mat)

np.set_printoptions(threshold=np.inf)
indices_c0 = [i for i, x in enumerate(model.labels_ == 0) if x]
indices_c1 = [i for i, x in enumerate(model.labels_ == 1) if x]
indices_c2 = [i for i, x in enumerate(model.labels_ == 2) if x]
indices_c3 = [i for i, x in enumerate(model.labels_ == 3) if x]

#print(indices_c0)
#print(indices_c1)
#print(indices_c2)
#print(indices_c3)

print("C0:")
for idx in indices_c0[:10]:
    print(first_few[idx])
print("==========================================")
print("C1:")
for idx in indices_c1:
    print(first_few[idx])
print("==========================================")
print("C2:")
for idx in indices_c2:
    print(first_few[idx])
print("==========================================")
print("C3:")
for idx in indices_c3:
    print(first_few[idx])

"""
print(model.labels_)
plt.figure()
plt.axes([0, 0, 1, 1])
for l, c in zip(np.arange(model.n_clusters), 'rgbk'):
    plt.plot(mat[model.labels_ == l].T, c=c, alpha=.5)
plt.axis('tight')
plt.axis('off')
plt.suptitle("AgglomerativeClustering(affinity=euclidean)", size=20)
plt.show()
"""