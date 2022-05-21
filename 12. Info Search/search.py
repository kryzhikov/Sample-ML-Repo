import random
import json
from build_index import Document
from nltk.stem import PorterStemmer
import re

# вот ссылка на датасет - https://www.kaggle.com/datasets/fabiochiusano/medium-articles
# там много-много разных текстиков с сайти Medium


# LENGTH = 192368
LENGTH = 10

# TODO: попробовать заменить на LancasterStemmer, сделать documents np.array

index = {}
documents = []


def unify_word(word):
    stemmer = PorterStemmer()  # 

    word = word.lower()
    word = re.sub(r'\B\W\b|\b\W\B|\B\W\B', ' ', word)
    word = stemmer.stem(word)
    return word


def build_index():
    # считывает сырые данные и строит индекс
    # данные я читаю из index.json, который надо создать, запустив файл build_index.py

    with open('12. Info Search\index.json') as f:
        info = json.load(f)

        for elem in info:
            documents.append(Document(**elem))

    for doc in documents:
        text = doc.get_text()
        for i, word in enumerate(text.split()):
            word = unify_word(word)
        
            if word not in index.keys():
                index[word] = []
            index[word].append(i)


def score(query, document):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    return random.random()


def find_union(arr1, arr2):
    arr1.sort()
    arr2.sort()
    union = []
    i = 0
    j = 0
    
    # ищем пересечение методом 2 указателей
    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:
            union.append(arr1)
            i += 1
            j += 1
        elif arr1[i] < arr2[j]:
            i += 1 
        elif arr1[i] > arr2[j]:
            j += 1 

    return union


def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    usage = []    
    for word in query.lower().split():
        word = unify_word(word)
        if word in index.keys():
            usage.append(index[word])

    if len(usage) == 0:
        return documents[:50]
    if len(usage) == 1:
        return usage[0][:50]

    candidates = find_union(usage[0], usage[1])
    for i in range(2, len(usage)):
        candidates = find_union(candidates, usage[i])
    
    docs = []
    for i in candidates[:50]:
        docs.append(documents[i])

    return docs


if __name__ == '__main__':
    build_index()
    print(len(retrieve('something of')))