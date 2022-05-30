import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
import re



class Document:
    def __init__(self, title, text, tf_idf):
        self.title = title
        self.text = text
        self.tf_idf = tf_idf

    def format(self, query):
        return [self.title,  re.sub(' ./ (\w){1}', '. \\1', self.text[:200] + ' ...').replace('-lrb-', '(').replace('-rrb-', ')').capitalize()]

indexes = {}
data = pd.DataFrame() 

def build_index():
    global data
    data = pd.read_csv('data.csv')
    for ind, biography in tqdm(data.iterrows()):
        for word in set(biography['stem_data'].split(' ')):
            if word not in indexes:
                indexes[word] = []
            indexes[word].append(ind)

def score(query, document):
    stemmer = SnowballStemmer(language='english')
    stem_query_word = [stemmer.stem(word) for word in query.strip().lower().split()]
    tf_idf = document.tf_idf
    scored = 0
    for word in stem_query_word:
        if word in document.title:
            scored += 1
        if word in tf_idf.columns:
            scored += tf_idf[word].iloc[0]
    return scored

def retrieve(query):
    if query == '':
        return []
    stemmer = SnowballStemmer(language='english')
    stem_query_word = [stemmer.stem(word) for word in query.strip().lower().split()]
    
    max_size_after_index = 1000
    after_index = []
    pages = []
    if (len(stem_query_word) == 1):
        after_index = indexes.get(stem_query_word[0], [])[0:max_size_after_index]
    else:
        for word in stem_query_word:
            pages.append(indexes.get(word, []))
        f_ind = 0
        s_ind = 0
        size = 0;
        while f_ind < len(pages[0]) and \
              s_ind < len(pages[1]) and \
              size < max_size_after_index:
            if pages[0][f_ind] < pages[1][s_ind]:
                f_ind += 1
            elif pages[0][f_ind] > pages[1][s_ind]:
                s_ind += 1
            else:
                after_index.append(pages[0][f_ind])
                f_ind += 1
                s_ind += 1
                size += 1
    if len(after_index) == 0:
        return []
    docs = []
    for ind in after_index:
        docs.append(data.iloc[ind]['stem_data'])
    tf_idfer = TfidfVectorizer()
    tf_idf = tf_idfer.fit_transform(docs)
    tf_idf = pd.DataFrame(tf_idf.toarray(), columns=tf_idfer.get_feature_names()).reset_index()
    response = []
    for ind in tf_idf['index'][0:min(20, len(tf_idf['index']))]:
        biography = data.iloc[after_index[int(ind)]]
        response.append(Document(biography['title'], biography['sents'], tf_idf.iloc[[int(ind)]]))
    return response

