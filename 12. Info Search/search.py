#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import multiprocessing
import numpy as np
from collections import Counter

import ir_datasets

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def process_text(text):
    """ Cleans and tokenizes a given text
    """
    text = text.lower()
    # tokenize
    text = [w for w in re.split(r'[&;,.\s]',text)]

    # remove numbers
    text = [w for w in text if not any(c.isdigit() for c in w)]
    
    # remove stop words
    stop = stopwords.words('english')
    text = [w for w in text if w not in stop]

    # remove empty tokens
    text = [w for w in text if len(w) > 0]

    # normalize
    stemmer = SnowballStemmer(language='english', ignore_stopwords=True)
    text = [stemmer.stem(w) for w in text]

    # remove words with only one letter
    text = [w for w in text if len(w) > 1]

    return text

class WIKIRDocument:
    def __init__(self, doc_id, text) -> None:
        self.id = doc_id
        self.text = text
        self.tokenized_text = process_text(text)

    def format(self, query):
        return ['Article '+self.id, self.text[:100] + ' ...']

def load_wikir_docs(n_docs):
    dataset = ir_datasets.load("wikir/en1k/training")
    wikir_docs = {}
    for doc_id, text in dataset.docs_iter()[:n_docs]:
        result = WIKIRDocument(doc_id, text)
        wikir_docs[doc_id] = result
    return wikir_docs

# load datset
n_docs = 5000
documents = load_wikir_docs(n_docs)
index = {}
TF_IDF = {}

def build_index():
    for doc_id, doc in documents.items():
        for word in set(doc.tokenized_text):
            if word not in index:
                index[word] = set()
            index[word].add(doc_id)

def build_tf_idf():
    # compute document frequency -- number of documents containing a term
    DF = {}
    for word in index.keys():
        DF[word] = len(index[word])

    # TF_IDF = {}
    n_docs = len(documents)
    for doc_id, doc in documents.items():
        counter = Counter(doc.tokenized_text)

        for token in np.unique(doc.tokenized_text):
            # calculate term frequency
            tf = counter[token]
            tf = 1 + np.log(tf)

            # calculate inverse document frequency
            idf = np.log((n_docs+1)/(DF[token]+1))

            TF_IDF[doc_id, token] = tf*idf

def score(query, document):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    query_words = process_text(query)
    score = 0
    for qw in query_words:
        try:
            score += TF_IDF[document.id, qw]
        except KeyError:
            pass

    return score

def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    query_words = process_text(query)
    candidates = []

    for qw in query_words:
        if qw in index:
            candidates.append(index[qw])

    # найди пересечение множеств документов соотвствующих каждому слову
    # в запросе
    if candidates:
        candidates = candidates[0].intersection(*candidates)
        return [documents[k] for k in candidates][:50]
    else:
        return candidates


if __name__ == "__main__":
    build_index()
    build_tf_idf()
    
    print(score("often", documents["0"]))
    print(score("often", documents["1"]))
    print(score("often", documents["2"]))