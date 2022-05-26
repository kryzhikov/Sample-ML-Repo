import random
import json
import re
import io
from unittest.mock import sentinel
import nltk
import numpy as np
import time
import logging

from tqdm import tqdm
from itertools import islice
from build_index import Document, unify_word
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# вот ссылка на датасет - https://www.kaggle.com/datasets/fabiochiusano/medium-articles
# там много-много разных текстиков с сайти Medium

# TODO: попробовать заменить на LancasterStemmer, сделать documents np.array

index = {}
words_frequency = {}
documents = []
vecs = None

logging.basicConfig(filename='12. Info Search\Logs\search.log', filemode='a', format='%(levelname)s %(message)s', level='DEBUG')

def load_vectors(fname, limit):
    fin = io.open(fname, 'r', encoding = 'utf-8', newline = '\n', errors = 'ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(islice(fin, limit), total = limit):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


def download_nltk():
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
    nltk.download('stopwords')


def get_index() -> None:
    # считывает сырые данные и строит индекс
    # данные я читаю из index.json, который надо создать, запустив файл build_index.py
    global documents
    logging.debug('Data reading started')

    start = time.time()
    with open('12. Info Search\documents.json') as f:
        info = json.load(f)

        for elem in info:
            documents.append(Document(**elem))

    documents = np.array(documents, dtype='object')

    logging.info(f'Reading data from file: {((time.time() - start) * 1000):.2f}ms')

    # vecs = load_vectors('12. Info Search\crawl-300d-2M.vec', 100000)
    download_nltk()

    with open('12. Info Search\index.json') as f:
        info = json.load(f)

        for elem in info.keys():
            index[elem] = info[elem]


def to_wordnet_tag(elem):
    if elem[1].startswith('J'):
        return elem[0], wn.ADJ
    if elem[1].startswith('V'):
        return elem[0], wn.VERB
    if elem[1].startswith('N'):
        return elem[0], wn.NOUN
    if elem[1].startswith('R'):
        return elem[0], wn.ADV
    return elem[0], wn.NOUN


def lemmatize(text, lemmatizer, tokenize='sent'):
    sw_en = stopwords.words('english')
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in sw_en])

    if tokenize == 'sent':
        tokenized = sent_tokenize(text)
    if tokenize == 'word':
        tokenized = word_tokenize(text)
    tagged = []

    # POS разметка
    for sent in tokenized:
        pos_tagged = pos_tag(sent.split())
        wordnet_taggs = []

        for elem in pos_tagged:
            wordnet_taggs.append(to_wordnet_tag(elem))

        tagged.append(wordnet_taggs)

    # лемматизация
    result = []
    for sent in tagged:
        lemmatized = []
        for word in sent:
            lemmatized.append(lemmatizer.lemmatize(word[0], pos=word[1]))
        
        result.append(' '.join(lemmatized))

    return ' '.join(result)


def stemmize(text, stemmer, tokenize='sent'):
    sw_en = stopwords.words('english')
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in sw_en])

    if tokenize == 'sent':
        tokenized = sent_tokenize(text)
    if tokenize == 'word':
        tokenized = word_tokenize(text)

    # стемминг
    result = []
    for sent in tokenized:
        lemmatized = []

        if tokenize == 'word':
            lemmatized.append(stemmer.stem(sent))
        if tokenize == 'sent':
            for word in sent.split():
                lemmatized.append(stemmer.stem(word))
        
        result.append(' '.join(lemmatized))

    return ' '.join(result)


def get_tf_idf_score(query, corpus_titles, corpus_texts, corpus_tags,
                         weights:list=[2, 1, 4]) -> int:

    vectorizer_titles = TfidfVectorizer(smooth_idf=True, norm='l2').fit([corpus_titles])
    tf_idf_title = vectorizer_titles.transform([corpus_titles]).toarray()

    vectorizer_texts = TfidfVectorizer(smooth_idf=True, norm='l2').fit([corpus_texts])
    tf_idf_text = vectorizer_texts.transform([corpus_texts]).toarray()

    vectorizer_tags = TfidfVectorizer(smooth_idf=True, norm='l2').fit([corpus_tags])
    tf_idf_tag = vectorizer_tags.transform([corpus_tags]).toarray()

    score = 0
    sections = list(zip( [vectorizer_titles, vectorizer_texts, vectorizer_tags],
                    [tf_idf_title, tf_idf_text, tf_idf_tag] ))

    for word in query.lower().split():
        for i, (vect, tf) in enumerate(sections):
            ind = np.argwhere(vect.get_feature_names_out() == word)

            if ind.size != 0:
                ind = ind[0][0]
                # print(ind, tf.shape)
            else:
                continue

            score += weights[i] * tf[0][ind]

    return score


def score(query:str, documents:list) -> list:
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    if query == '':
        return zip(documents, [0 for _ in range(len(documents))])

    logging.debug(f'Scoring started')

    # TODO: word2vec
    start = time.time()
    lemmatizer = WordNetLemmatizer()
    stemmizer = PorterStemmer()

    corpus_titles = [stemmize(doc.title, stemmizer) for doc in documents]
    corpus_texts = [stemmize(doc.text, stemmizer) for doc in documents]
    corpus_tags = [stemmize(doc.str_tags(), stemmizer) for doc in documents]

    query = stemmize(query, stemmizer, tokenize='word')
    logging.info(f'Lemmatizing all docs: {((time.time() - start) * 1000):.2f}ms')

    start = time.time()
    tf_idf = []
    for title, text, tag in zip(corpus_titles, corpus_texts, corpus_tags):
        tf_idf.append( get_tf_idf_score(query, title, text, tag) )
    logging.info(f'Counting all score took: {((time.time() - start) * 1000):.2f}ms')

    return zip(documents, tf_idf)


def find_union(arr1:list, arr2:list) -> list:
    arr1.sort()
    arr2.sort()
    union = []
    i = 0
    j = 0
    
    # ищем пересечение методом 2 указателей
    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:
            union.append(arr1[i])
            i += 1
            j += 1
        elif arr1[i] < arr2[j]:
            i += 1 
        elif arr1[i] > arr2[j]:
            j += 1 

    return union


def sort_by_count(query:str, documents:list)->list:
    counter = []

    for doc in documents:
        result = 0

        for word in query.split():
            word = unify_word(word)
            result += doc[0].get_text().lower().count(word)

        counter.append((result, doc))
    
    counter.sort(key=lambda x: x[0], reverse=True)
    counter = np.array(counter)

    return counter[:, 1]


def get_docs_by_id(ids:list) -> list:
    logging.debug('Getting docs by id')

    start = time.time()
    docs = []
    for i in ids:
        docs.append(documents[i])
    logging.info(f'Getting docs took: {((time.time() - start) * 1000):.2f}ms')

    return docs


def retrieve(query:str) -> list:
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    logging.debug('Retrieving 50 docs started')

    start = time.time()
    usage = []    
    for word in query.lower().split():
        word = unify_word(word)
        if word in index.keys():
            usage.append(list(index[word]))
    logging.info(f'Find all documents containing words: {((time.time() - start) * 1000):.2f}ms')

    if len(usage) == 0:
        return documents[:50], [i for i in range(0, 50)]
    if len(usage) == 1:
        documents = zip(get_docs_by_id(usage[0]), usage[0]) 
        documents = sort_by_count(query, documents)
        return zip(*list(documents[:50]))

    start = time.time()
    candidates = find_union(usage[0], usage[1])
    for i in range(2, len(usage)):
        candidates = find_union(candidates, usage[i])
    logging.info(f'Set union: {((time.time() - start) * 1000):.2f}ms')
    
    documents = list(zip(get_docs_by_id(candidates[:100]), candidates[:100]))
    documents = sort_by_count(query, documents)

    return zip(*list(documents[:50]))


def pretify(text):
    return re.sub(r'\n', r'<br />', text)


def get_doc(id:int) -> Document:
    return documents[id]

if __name__ == '__main__':
    query = 'mental health'
    get_index()

    documents, ids = retrieve(query)

    scored = score(query, documents)
    scored = list(zip(scored, ids))

    scored = sorted(scored, key=lambda doc: -doc[0][1])
    print(scored[:5])