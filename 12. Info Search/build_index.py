import concurrent.futures as cf
import multiprocessing as mp
from multiprocessing import Array
import pandas as pd
import re
import json


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
        return self.title + self.text + ' '.join(self.tags)

    def str_tags(self):
        return ' '.join(self.tags)

    def format(self, query):
        # возвращает пару тайтл-текст-url, отформатированную под запрос
        return [self.title, self.text[:150] + ' ...', self.url]

    def __dict__(self):
        return {'title': self.title, 'text': self.text, 'url': self.url, 'tags': self.tags}

    def __str__(self):
        return f'{self.title[:10]}... {self.text[:10]}... {self.url}, {self.tags[:3]}'


LENGTH = 10

def str_to_list(str):
    elements = str.split(',')
    arr = []

    for elem in elements:
        arr.append( re.sub(r'[\[\'\]]+', '', elem) )

    return arr


def add_new_document(info, results):
    results.put( Document(
        info['title'], info['text'], info['url'], str_to_list(info['tags'])
    ))


def template(info):
    return Document('title', 'text', 'url', [str(info)])


def save_index(index):
    with open('12. Info Search\index.json', 'w') as f:
        pages = []
        for elem in index:
            pages.append(elem.__dict__())

        data = json.dumps(pages)
        f.write(data)


def build():
    df = pd.read_csv('12. Info Search\medium_articles.csv')

    results = mp.Queue(LENGTH)
    # index = []
    # for i in range(LENGTH):
        # index.append( add_new_document(df.iloc[i]))


    with cf.ProcessPoolExecutor() as executor:
        executor.map(add_new_document, [(df.iloc[i], results) for i in range(LENGTH)])

    # with cf.ProcessPoolExecutor() as executor:
    #     res = [executor.submit(template, i) for i in range(LENGTH)]

    #     for f in cf.as_completed(res):
    #         print(f.result())

    print(results.qsize())
    # save_index(index)


if __name__ == '__main__':
    build()