import concurrent.futures as cf
import multiprocessing as mp
from multiprocessing import Array, Lock
from nltk.stem import PorterStemmer
import pandas as pd
import re
import json
import time
import logging


class Document:
    def __init__(self, title, text, url, tags):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text
        self.url = url
        self.tags = self.lower_array(tags)
    
    @staticmethod
    def lower_array(arr):
        arr = list(map(str.lower, arr))
        return arr

    def get_text(self):
        return str(self.title) + str(self.text) + ' '.join(self.tags)

    def str_tags(self):
        return ' '.join(self.tags)

    def format(self, query):
        # возвращает пару тайтл-текст-url, отформатированную под запрос
        return [self.title, self.text[:150] + ' ...', self.url]

    def __dict__(self):
        return {'title': self.title, 'text': self.text, 'url': self.url, 'tags': self.tags}

    def __str__(self):
        return f'{self.title[:10]}... {self.text[:10]}... {self.url}, {self.tags[:3]}'


logging.basicConfig(filename='12. Info Search\Logs\search.log', filemode='a', format='%(levelname)s %(message)s', level='DEBUG')
# LENGTH = 192368
LENGTH = 50000

def str_to_list(str):
    elements = str.split(',')
    arr = []

    for elem in elements:
        arr.append( re.sub(r'[\[\'\]]+', '', elem) )

    return arr


def add_new_document(info):
    return Document(
            info['title'], info['text'], info['url'], str_to_list(info['tags'])
    )


def template(info):
    return Document('title', 'text', 'url', [str(info)])


def unify_word(word:str) -> str:
    stemmer = PorterStemmer() 

    word = word.lower()
    word = re.sub(r'\B\W\b|\b\W\B|\B\W\B', ' ', word)
    word = stemmer.stem(word)
    return word


def save_index(index):
    with open('12. Info Search\index.json', 'w') as f:
        pages = {}

        for key in index.keys():
            pages[key] = list(index[key])

        data = json.dumps(pages)
        f.write(data)


def save_docs(docs):
    with open('12. Info Search\documents.json', 'w') as f:
        pages = []

        for doc in docs:
            pages.append(doc.__dict__())

        data = json.dumps(pages)
        f.write(data)


def build():
    df = pd.read_csv('12. Info Search\medium_articles.csv')

    start = time.time()

    documents = []
    for i in range(LENGTH):
        documents.append( add_new_document(df.iloc[i]))

    print(f'{(time.time() - start):.2f} seconds')

    # with cf.ProcessPoolExecutor() as executor:
        # executor.map(add_new_document, [(df.iloc[i], results) for i in range(LENGTH)])

    # processes = []
    # NUMBER_OF_PROCESSES = 4

    # for _ in range(NUMBER_OF_PROCESSES):
    #     p = mp.Process(target=add_new_document, args=(df.iloc[0], results))
    #     p.start()
    #     print('Process started')
    #     processes.append(p)

    # for pr in processes:
    #     pr.join()
    #     print('Joining...')

    # print(results.qsize())
    # for i in range(LENGTH):
    #     print(results.get())

    # print('Finished')
    
    save_docs(documents)

    index = {}
    start = time.time()

    for i, doc in enumerate(documents):
        text = doc.get_text()
        for word in text.split():
            word = unify_word(word)
        
            if word not in index.keys():
                index[word] = set()

            index[word].add(i)
    logging.info(f'Building index: {((time.time() - start) * 1000):.2f}ms')

    save_index(index)


if __name__ == '__main__':
    build()