from matplotlib.pyplot import title
import preprocessing as pr
import pandas as pd
import re
import numpy as np

class Document:
    
    def __init__(self, title, text):
        
        self.title = title
        self.text = text
    

    def format(self, query):
        
        return [self.title, self.text + ' ...']


index = {}

data = pd.read_csv('/Volumes/Transcend/Documents/Tinkoff/ML/Sample-ML-Repo/12. Info Search/search/data.csv')


def build_index():
    
    overviews = data['overview'].values
    titles = data['title'].values
    
    for i, text in enumerate(overviews): 
        text = pr.nltk(text)
        
        for word in set(re.compile('[A-z]+').findall(text)):
            if word not in index: index[word] = []
            index[word].append(i)

    for i, text in enumerate(titles): 
        text = pr.nltk(text)
        
        for word in set(re.compile('[A-z]+').findall(text)):
            if word not in index: index[word] = []
            index[word].append(i)


def score(query, document, query_neighbours=['']):
    
    q = pr.vectorize(pr.nltk(query))
    q_n = pr.vectorize(pr.nltk(' '.join(query_neighbours)))
    score = pr.score(q + q_n / 5).dist(pr.vectorize(pr.nltk(document.text)) + pr.vectorize(pr.nltk(document.title)))
    
    return 1 / score


def TwoSetIntersections(set1, set2=[]):

    i = j = 0
    res = []

    if len(set2) == 0: return set1[:1000]
    if len(set1) == 0: return set2[:1000]

    while ((i < len(set1)) or (j < len(set2))) and len(res) < 1000:

        if set1[min(i, len(set1) - 1)] < set2[min(j, len(set2) - 1)]:
            i += 1
            if i >= len(set1) - 1: j += 1

        elif set1[min(i, len(set1) - 1)] > set2[min(j, len(set2) - 1)]:
            j += 1
            if j >= len(set2) - 1: i += 1

        else:
            res.append(set1[i])
            i += 1
            j += 1

    return res


def intersections(set):
    
    if len(set) == 0: return [] 
    
    res = TwoSetIntersections(set[0])
    
    for i in set[1:]:
        res = TwoSetIntersections(res, i)

    return res


def retrieve(query):
    
    query = re.compile('[A-z]+').findall(query.lower())
    
    query_neighbours = np.array([pr.get_k_nearest_neighbors(pr.vectorize(word), 10) for word in query]).flatten()
    query_neighbours = [re.compile('[A-z]+').findall(n.lower()) for n in query_neighbours]
    while query_neighbours.count([]) != 0: query_neighbours.remove([])
    query_neighbours = set([n[0] for n in query_neighbours]).difference(set(query))
    
    files = []
    
    for word in query:
        if word in index:
            if index[word] in files:
                continue
            files.append(index[word])

    docs = intersections(files)
    
    if len(docs) == 0:
        return [Document(title='Nothing was found', text='Try another query')], []
    
    response_docs = []

    for i in docs[:1000]:
        response_docs.append([data['title'].values[i], data['overview'].values[i]])

    return [Document(title, text) for title, text in response_docs], query_neighbours