import random
import pandas as pd
import re
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenize(text: str, regex="[А-Яа-яёA-z]+") -> list: # будем токенизировать с помощью регулярки
    regex = re.compile(regex)
    tokens = regex.findall(text)

    return tokens

class Document:
    def __init__(self, title, text):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text + ' ...']

index = {}

df = pd.read_csv(r'C:\Users\alina\Desktop\df_cleaned.csv') #можно скачать по ссылке с каггла и обработать, на гит не влезает
df.dropna()
df = df.sort_values(by='date', ascending=False) #сортим по дате, нам же важны свежие новости
df.reset_index(inplace=True)

def build_index():
     for i, text in enumerate(df.clean1):
        if isinstance(text, str):
            for word in set(text.split()):
                if word.lower() not in index:
                    index[word] = []
                index[word].append(i)      


def tfidf(a, b, vectorizer=TfidfVectorizer()):
    tfidf = vectorizer.fit_transform([a, b])
    return ((tfidf * tfidf.T).toarray())[0, 1]

def intersection(x):
    ans = inter(x[0], x[1])
    for i in range(2, len(x)):
        ans = inter(x[i], ans)
    return ans

def inter(a, b):
    out = []
    i = 0
    j = 0
    a = np.array(a)
    a = a.ravel()
    b = np.array(b)
    b = b.ravel()
    #print(a[:10], b[:10])
    while (i < len(a)) and (j < len(b)):
        if a[i] > b[j]:
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            out.append(a[i])
            i += 1
            j += 1
    return out

def score(query, document):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    if query == '':
        return 0.0
    return tfidf(query, document.text)
    #return random.random()

def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)

    if query == '':
        return [Document(title="Not found", text="...")]

    query = query.lower()
    query = [i.strip(string.punctuation) for i in query.split()]
    #print(query)
    #query = " ".join(query)

 
    candidates = []
    candidates1 = []

    for i in query:
        tmp = []
        if  i in index:
            if index[i] in tmp:
                continue
            tmp.append(index[i])
        candidates.append(tmp)

    #print(len(candidates))

    for i in range(len(candidates)):
        candidates[i].sort()

    if len(candidates) == 0:
        return [Document(title="Not found", text="...")]

    if len(candidates) >1:
        candidates1 = intersection(candidates) 

    if len(candidates) ==1:
        candidates1 = candidates

    #print(candidates1)
    candidates2 = []

    #if len(candidates1) < 30:
    #    tmp = 0
    #    for i in range(len(candidates)):
    #        for j in range(len(candidates[i])):
    #            candidates2.append(candidates[i][j])
    #            tmp+=1
    #            if tmp > 29:
    #                break
    #            
    #        if tmp > 29:
    #            break
      
    
    hlp = np.array(candidates1)
    hlp = hlp.ravel()[:15]
    docs = []
    for i in hlp:
        docs.append([str(df.loc[i, 'title']), str(df.loc[i, 'content'])])


    return [Document(title, text) for title, text in docs]
