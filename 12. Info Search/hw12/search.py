import random
from collections import defaultdict
from sklearn.metrics import ndcg_score
import numpy as np
import editdistance


class Document:
    def __init__(self, text, artist, song, seq):
        self.text = text
        self.artist = artist
        self.song = song
        self.seq = seq

    def format(self):
        return [self.artist + ' - ' + self.song, self.seq + ' ...']


index = defaultdict(list)


def build_index(x):
    documents = x['text']
    for ind, doc in enumerate(documents):
        for word in set(doc.split(' ')):
            index[word].append(ind)


def score(query, document):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    #ndcg_score(np.random.randint(1, 6, size=(1, 5)), np.random.randint(1, 6, size=(1, 5)))
    return 1.0 - editdistance.eval(query, document.text) / (1.0 * max(len(query), len(document.text)))


def retrieve(query, x):
    # возвращает начальный не бесконечный список релевантных документов
    all_list, next_list, candidates = [], [], []

    for word in set(query.lower().split(' ')):
        if word in index:
            all_list = index[word]
            break

    for word in set(query.lower().split(' ')):
        if word in index:
            next_list = index[word]
            all_list = list(set(all_list) & set(next_list))

    for i in all_list:
        candidates.append(Document(x['text'].iloc[i], x['artist'].iloc[i], x['song'].iloc[i], (x['seq'].iloc[i])[:90]))
    return candidates[:50]