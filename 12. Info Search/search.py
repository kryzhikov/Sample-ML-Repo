import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from functools import reduce

sw_eng = set(stopwords.words('english'))
stemmer = SnowballStemmer(language='english')

class Document:
    def __init__(self, id, title, desc, author_proc, desc_proc, genre_proc, title_proc):
        # можете здесь какие-нибудь свои поля подобавлять
        self.id = id
        self.title = title
        self.desc = desc
        self.author_proc = author_proc
        self.desc_proc = desc_proc
        self.genre_proc = genre_proc
        self.title_proc = title_proc
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.desc[:100] + ' ...']

index = []
title_invert_index = {}
text_invert_index = {}
genre_invert_index = {}
author_invert_index = {}

def build_index():
    global index, title_invert_index, text_invert_index, \
        genre_invert_index, author_invert_index
    # считывает сырые данные и строит индекс
    df = pd.read_csv('../Downloads/search_data.csv')
    for i, row in df.iterrows():
        index.append(Document(i, *row))

    title_invert_index = build_invert_index(df['title_proc'].tolist())
    text_invert_index = build_invert_index(df['desc_proc'].tolist())
    genre_invert_index = build_invert_index(df['genre_proc'].tolist())
    author_invert_index = build_invert_index(df['author_proc'].tolist())


def build_invert_index(lst):
    invert_index = {}
    for i in range(len(lst)):
        for word in str(lst[i]).split():
            if word not in invert_index.keys():
                invert_index[word] = [i]
            else:
                invert_index[word].append(i)
    return invert_index

def process_text(text):
    # remove punctuation
    for punctuation in string.punctuation:
        text = str(text).replace(punctuation, '')
    # tokenize
    expr = r'[^(\w.\w)\w\s]'
    parser = re.compile(expr)
    text = parser.sub(r'', text).split()
    # to lower
    text = [word.lower() for word in text]
    # delete stop-words
    text = [word for word in text if word not in sw_eng]
    # stemming
    return ' '.join([stemmer.stem(word) for word in text])

def score(query, document):
    global index
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    processed_query = process_text(query)
    title_list = [document.title_proc, processed_query]
    text_list = [document.desc_proc, processed_query]
    genre_list = [document.genre_proc, processed_query]
    author_list = [document.author_proc, processed_query]
    res = []
    for lst in [title_list, text_list, genre_list, author_list]:
        try:
            vectorizer = TfidfVectorizer().fit_transform(lst)
            vectors = vectorizer.toarray()
            res.append(cos_similarity(vectors[0], vectors[1]))
        except ValueError:
            res.append(0)
    return 0.3 * res[0] + 0.2 * res[1] + 0.2 * res[2] + 0.3 * res[3]

def cos_similarity(vector_1, vector_2):
    vector_1 = vector_1.reshape(1, -1)
    vector_2 = vector_2.reshape(1, -1)
    return cosine_similarity(vector_1, vector_2)[0][0]


def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    global index, title_invert_index, text_invert_index, \
        genre_invert_index, author_invert_index
    processed_query = process_text(query).split()
    words_indexes = []
    for word in processed_query:
        if word in title_invert_index.keys():
            words_indexes.append(title_invert_index[word])
        if word in text_invert_index.keys():
            words_indexes.append(text_invert_index[word])
    if len(words_indexes) == 0:
        return []
    else:
    # не хватило времени на способ с двумя указателями, поэтому так:
        words_indexes = reduce(lambda x, y: x + y, words_indexes)
        # отсортирую документы, по кол-ву совпавших слов с запросом и выведу первые 50
        candidates = []
        cnt = Counter(words_indexes)
        sorted_cnt = list(dict(sorted(cnt.items(), key=lambda item: item[1])).keys())
        for i in range(51):
            candidates.append(index[sorted_cnt[i]])

        return candidates[:50]


import time
start_time = time.time()
build_index()
print("Построение индекса заняло --- %s seconds ---" % (time.time() - start_time))
