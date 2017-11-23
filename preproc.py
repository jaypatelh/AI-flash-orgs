import string
import json
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
from nltk import ngrams
from nltk import tokenize
import enchant
import sys
import operator

stop = stopwords.words("english")
stemmer = SnowballStemmer("english")
def clean(word):
    word = word.lower()
    if word in stop: # or len(word) <= 2:
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

translator = str.maketrans('', '', string.punctuation)  #remove punctuation
vocab = {}
descriptions = []
titles = []
with open('mobile_test/sent_data.txt', 'w') as sf:
    for i in range(1,51):
        with open('freelancer-com-scrape/data' + str(i) + '.json') as jf:
            jd = json.load(jf)
            for d in jd:
                #combined = d['desc'] + ' ' + d['title']
                combined = d['desc']
                cd = combined.replace("[url removed, login to view]", "")            
                # print(cd)
                sentences = tokenize.sent_tokenize(cd)
                sentences = [sent.translate(translator) for sent in sentences]
                # print(sentences)  #sentences is an array of sentences for each "doc". 
                stem_sentences = [" ".join(filter(None, [clean(word) for word in sentence.split(" ")])) for sentence in sentences]
                # print(stem_sentences)

                stem_title = " ".join(filter(None, [clean(word) for word in d['title'].split(" ")]))
                sf.write("%s\n" % stem_title)
                for sent in stem_sentences:
                    sf.write("%s\n" % sent)
                
                # vcb, split_cd = unigrams(cd)
                # add_dict_counts(vocab, vcb)
                # descriptions.append(split_cd)
                
                # titles.append(d['title'])

# vocab, all_counts = filter_below_threshold(vocab)
#for k in sorted(all_counts.keys(), key=lambda x: int(x)):
#    print(k, ' ', all_counts[k])
# vi = list(vocab.keys())
# vocab_size = len(vi)
print 