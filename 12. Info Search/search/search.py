import random

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class Document:
    def __init__(self, title, text, tfidf):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text
        self.tfidf = tfidf

    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text + ' ...']


index = {}
df = pd.DataFrame()


def build_index():
    global df
    df = pd.read_csv('movies_new.csv')
    for i, movie in df.iterrows():
        for word in set(movie['word_concat'].split(' ')):
            if word not in index:
                index[word] = []
            index[word].append(i)


def score(query, document):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    stemmer = SnowballStemmer(language='english')
    words = [stemmer.stem(word) for word in query.split()][:2]
    tfidf = document.tfidf
    score = 0
    for word in words:
        if word in tfidf.columns:
            score += tfidf[word]
    return score


def retrieve(query):
    if query == '':
        return []
    stemmer = SnowballStemmer(language='english')
    pages = []
    words = [stemmer.stem(word) for word in query.split()][:2]
    res = []
    max_size = 100
    if len(words) == 1:
        pages = index.get(words[0], [])[:max_size]
        for i in pages:
            res.append(i)
    else:
        for word in words:
            pages.append(index.get(word, []))
        i = 0
        j = 0
        l_first = len(pages[0])
        l_sec = len(pages[1])
        size = 0

        while i < l_first and j < l_sec and size < max_size:
            if pages[0][i] < pages[1][j]:
                i += 1
            elif pages[0][i] > pages[1][j]:
                j += 1
            else:
                res.append(pages[0][i])
                i += 1
                j += 1
                size += 1

    corpus = []

    for i in res:
        corpus.append(df.iloc[i]['word_concat'])

    idf_vectorizer = TfidfVectorizer()
    Y = idf_vectorizer.fit_transform(corpus)
    tfidf = pd.DataFrame(Y.toarray(), columns=idf_vectorizer.get_feature_names()).reset_index()
    response = []
    l_tfidf = len(tfidf.index)
    max_resp_size = 10
    for idx in tfidf.index[:min(max_resp_size, l_tfidf)]:
        movie = df.iloc[res[idx]]
        response.append(Document(movie['title'], movie['overview'], tfidf.iloc[[idx]]))
    return response
