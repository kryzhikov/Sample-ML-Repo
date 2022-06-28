from nltk.corpus import stopwords
sw_eng = set(stopwords.words('english'))

def nltk(text):

    clean_sent = ' '.join([word for word in re.compile('[A-z]+').findall(text.lower()) if not word in sw_eng])

    return clean_sent.lower()


import io
import numpy as np
from itertools import islice
import re


def __load_vectors__(fname, limit):

    fin = io.open(fname, 'r', encoding = 'utf-8', newline = '\n', errors = 'ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in islice(fin, limit):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))

    return data


vecs = __load_vectors__('/Volumes/Transcend/Documents/Tinkoff/ML/Sample-ML-Repo/12. Info Search/search/crawl-300d-2M.vec', 100000)
vecs2 = __load_vectors__('/Volumes/Transcend/Documents/Tinkoff/ML/Sample-ML-Repo/12. Info Search/search/crawl-300d-2M.vec', 10000)


def get_k_nearest_neighbors(vec, k):
    return list(zip(*sorted(list(map(lambda key: (np.linalg.norm(vec - vecs2[key]), key), vecs2.keys())))))[1][:k]


def vectorize(text):

    text_vec = np.zeros(300)
    for word in set(re.compile('[A-z]+').findall(text.lower())):
        try:
            text_vec += vecs[word]
        except Exception:
            pass

    return text_vec / max(1, len(set(re.compile('[A-z]+').findall(text.lower()))))


from sklearn.metrics.pairwise import cosine_distances


class score:

    def __init__(self, q=''):
        self.q = q

    def dist(self, text):

        return cosine_distances(self.q.reshape(1, -1), text.reshape(1, -1)).flatten()[0]