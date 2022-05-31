import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import reduce


class Document:
    def __init__(self, id, title, content, title1, content1):
        self.id = id
        self.title = title
        self.content = content
        self.title1 = title1
        self.content1 = content1
    
    def format(self, query):
        return [self.title, self.content[:200] + ' ...']

title1 = {}
content1 = {}
index = []

def build_index():
    global index, title1, content1
    df = pd.read_csv('info_search.csv')
    for i, j in df.iterrows():
        index.append(Document(i, *j))

    slov = {}
    for j in range(len(df['title1'].tolist())):
        for word in str(df['title1'].tolist()[j]).split():
            if word not in slov.keys():
                slov[word] = [j]
            else:
                slov[word].append(j)
    title1 = slov

    slov = {}
    for j in range(len(df['content1'].tolist())):
        for word in str(df['content1'].tolist()[j]).split():
            if word not in slov.keys():
                slov[word] = [j]
            else:
                slov[word].append(j)
    content1 = slov

def score(query, document):
    res = []
    for i in [[document.title1, query], [document.content1, query]]:
        try:
            i_tv = TfidfVectorizer().fit_transform(i)
            res.append(((i_tv * i_tv.T).toarray())[0, 1])
        except ValueError:
            res.append(0)
    return 0.3*  res[0] + 0.2 * res[1]

def f2(a, b):
  cf = []
  for i in a:
      if i in cf:
          continue
      for j in b:
          if i == j:
              cf.append(i)
              break
  return cf

def f3(L):
    def R(a, b, seen=set()):
        a.update(b & seen)
        seen.update(b)
        return a
    return reduce(R, map(set, L), set())

def retrieve(query):
    global index, title1, content1
    query = query.split()
    c = []
    for word in query:
        if word in title1.keys():
            c.append(title1[word])
        if word in content1.keys():
            c.append(content1[word])
    if len(c) == 0:
        return [Document(id = '',  title = 'Sorry, nothing was found on the request(((', content = '', title1 = '', content1 = '')]

    elif len(c) == 1:
        c2 = []
        c2.append(c[0])
        c2 = reduce(lambda x, y: x + y, c2)
        result = []
        c2_count = Counter(c2)
        sorted_cnt = list(dict(sorted(c2_count.items(), key=lambda item: item[1])).keys())
        for i in range(60):
            result.append(index[sorted_cnt[i]])
        return result

    elif len(c) == 2:
        c2 = []
        c2.append(f2(c[0], c[1]))
        c2 = reduce(lambda x, y: x + y, c2)
        result = []
        c2_count = Counter(c2)
        sorted_cnt = list(dict(sorted(c2_count.items(), key=lambda item: item[1])).keys())
        for i in range(60):
            result.append(index[sorted_cnt[i]])
        return result

    else:
        c2 = []
        c2.append(f3(c))
        c2 = reduce(lambda x, y: x + y, c2)
        result = []
        c2_count = Counter(c2)
        sorted_cnt = list(dict(sorted(c2_count.items(), key=lambda item: item[1])).keys())
        for i in range(60):
            result.append(index[sorted_cnt[i]])
        return result